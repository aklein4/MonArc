from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.models.stablelm.modeling_stablelm import StableLmDecoderLayer, StableLmAttention

from annelid.configuration_annelid import AnnelidConfig


def _attn_forward(self, hidden_states, *args, **kwargs):
    return (hidden_states, None, None)
StableLmAttention.forward = _attn_forward


class AnnelidPreTrainedModel(PreTrainedModel):
    config_class = AnnelidConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StableLmDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data = torch.randn_like(module.weight.data) * std
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
        elif isinstance(module, nn.Embedding):
            module.weight.data = torch.randn_like(module.weight.data) * std


class AnnelidModel(AnnelidPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`StableLmDecoderLayer`]

    Args:
        config: AnnelidConfig
    """

    def __init__(self, config: AnnelidConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # error checking
        assert not (config.is_prefix_lm and config.is_quasi_lm), "Cannot be both prefix and quasi language model!"
        if config.is_prefix_lm or config.is_quasi_lm:
            assert config._attn_implementation == 'sdpa', "Prefix and quasi language models only support sdpa attention (require custom attention masks)"

        # save custom config info
        self.is_prefix_lm = config.is_prefix_lm
        self.is_quasi_lm = config.is_quasi_lm
        self.segment_size = config.segment_size
        self.use_segment_embeds = config.use_segment_embeds

        # Standard weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # extra parameters for prefix/quasi LM
        self.segment_embeds = None
        self.prefix_embeds = None
        if self.use_segment_embeds:
            self.segment_embeds = nn.Embedding(self.segment_size, config.hidden_size)
        if self.is_prefix_lm or self.is_quasi_lm:
            self.prefix_embeds = nn.Embedding(2, config.hidden_size)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def _get_tokens(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None
    ):
        # batch_size, seq_length = input_ids.size()

        # double if this is a quasi LM
        if self.is_quasi_lm:
            input_ids = torch.cat([input_ids, input_ids], dim=1)
            seq_length *= 2

        # get the id tokens
        tokens = self.embed_tokens(input_ids)
        return tokens

        # add segment embeddings
        if self.use_segment_embeds:
            ar = torch.arange(seq_length, device=input_ids.device, dtype=input_ids.dtype)
            segment_ids = ar % self.segment_size
            segment_embs = self.segment_embeds(segment_ids).unsqueeze(0)

            tokens = tokens + segment_embs

        # prefix embedddings
        if self.is_prefix_lm:
            ar = torch.arange(seq_length, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
            prefix_ids = (ar < prefix_length).to(input_ids.dtype)
            prefix_embs = self.prefix_embeds(prefix_ids)

            tokens = tokens + prefix_embs

        # quasi embeddings
        if self.is_quasi_lm:
            quasi_ids = torch.zeros_like(input_ids)
            quasi_ids[:, :seq_length//2].fill_(1)
            quasi_embs = self.prefix_embeds(quasi_ids)

            tokens = tokens + quasi_embs

        return tokens
    

    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None
    ):
        # batch_size, seq_length = input_ids.shape
    
        # prefix lm is bidirectional for prompt
        if self.is_prefix_lm:

            # get standard mask
            mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)
            mask = torch.triu(mask, diagonal=1)

            # apply prefix
            ar = torch.arange(seq_length, dtype=torch.long)
            p = torch.maximum(ar[:, None], ar[None, :]).unsqueeze(0)
            mask = torch.where(p < prefix_length[:, None, None], torch.zeros_like(mask), mask)
        
        # quasi LM is bidirectional for segments
        elif self.is_quasi_lm:
            if seq_length % self.segment_size != 0:
                raise NotImplementedError("Quasi LM only supports inputs that are tiled by segment size")
            n_segments = seq_length // self.segment_size

            # self attending segments
            nw = torch.ones(batch_size, n_segments, n_segments, dtype=torch.bool)
            nw = torch.triu(nw, diagonal=1)
            nw = torch.repeat_interleave(nw, self.segment_size, dim=1)
            nw = torch.repeat_interleave(nw, self.segment_size, dim=2)

            # segments for cross attention
            sw = torch.ones(batch_size, n_segments, n_segments, dtype=torch.bool)
            sw = torch.triu(sw, diagonal=0)
            sw = torch.repeat_interleave(sw, self.segment_size, dim=1)
            sw = torch.repeat_interleave(sw, self.segment_size, dim=2)

            # empty
            ne = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)

            # auto regressive within segments
            se = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)
            se = torch.triu(se, diagonal=1)
            se = torch.logical_xor(se, ~sw)

            # send ouf as bias
            mask = torch.cat(
                [
                    torch.cat([nw, ne], dim=2),
                    torch.cat([sw, se], dim=2)
                ],
                dim=1
            )
        
        # use standard mask for standard LM
        else:
            mask = None
        
        if mask is not None:
            mask = ~mask # flip for sdpa attention
            mask = mask.unsqueeze(1) # head dim
            mask = mask.to(input_ids.device)

        return mask


    def get_position_ids(
        self,
        input_ids: torch.LongTensor
    ):
        # batch_size, seq_length = input_ids.shape

        return None

        # standard positions
        pos = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)

        # quasi lm repeats the sequence
        if self.is_quasi_lm:
            return torch.cat([pos, pos], dim=1)

        # otherwise return the standard positions
        return pos


    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None
    ) -> torch.Tensor:
        # batch_size, seq_length = input_ids.shape

        # error checking
        if not self.is_prefix_lm:
            assert prefix_length is None, "Prefix length only used for prefix LM"
        else:
            assert prefix_length is not None, "Prefix length required for prefix LM"
            assert isinstance(prefix_length, torch.Tensor), "Prefix length must be a tensor"
            assert tuple(prefix_length.shape) == (batch_size,), "Prefix length must have shape (batch_size,)"

        # get inputs
        hidden_states = self._get_tokens(input_ids, prefix_length)
        return hidden_states
        mask = self._get_mask(input_ids, prefix_length)
        pos = self.get_position_ids(input_ids)

        for decoder_layer in self.layers:

            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=mask,
                position_ids=pos,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]

        # if quasi, remove encoding portion
        if self.is_quasi_lm:
            hidden_states = hidden_states[:, -seq_length:]

        hidden_states = self.norm(hidden_states)

        return hidden_states


# Copied from transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM with PERSIMMON->STABLELM,Persimmon->StableLm
class AnnelidLMModel(AnnelidPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with LLAMA->STABLELM,Llama->StableLm
    def __init__(self, config):
        super().__init__(config)
        self.model = AnnelidModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model


    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None,
    ) -> torch.Tensor:

        out = self.model(
            input_ids=input_ids,
            prefix_length=prefix_length
        )

        logits = self.lm_head(out)
        logits = F.log_softmax(logits, dim=-1)

        return logits
