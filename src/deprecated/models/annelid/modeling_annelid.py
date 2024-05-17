from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
except ImportError:
    print("WARNING: torch_xla not installed, cannot use gradient checkpointing")

from transformers.modeling_utils import PreTrainedModel
from transformers.models.stablelm.modeling_stablelm import StableLmDecoderLayer

from models.annelid.configuration_annelid import AnnelidConfig
from utils.data_utils import DotDict
from utils.logging_utils import log_print


class AnnelidPreTrainedModel(PreTrainedModel):

    # all StableLM/renamed
    config_class = AnnelidConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StableLmDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_sdpa = True

    # from StableLM
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    # converted from torch to torch xla
    def xla_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        gradient_checkpointing_func = functools.partial(xla_checkpoint_fn, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


    def _xla_set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """ Set gradient checkpointing for base model and submodules. """
        
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )


class AnnelidModel(AnnelidPreTrainedModel):

    def __init__(self, config: AnnelidConfig):
        """
        StableLM-based language model.
        Supports:
        - Causal Decoder Model
        - Prefix Language Model
        - Quasi-Causal Language Model

        Args:
            config: AnnelidConfig
        """
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # custom config info
        self.is_prefix_lm = config.is_prefix_lm
        self.is_quasi_lm = config.is_quasi_lm
        self.segment_size = config.segment_size
        self.use_segment_embeds = config.use_segment_embeds

        # error checking
        assert not (config.is_prefix_lm and config.is_quasi_lm), "Cannot be both prefix and quasi language model!"
        if config.is_prefix_lm or config.is_quasi_lm:
            assert config._attn_implementation != 'flash_attention_2', "Prefix and quasi language models do not support flash attention (require custom attention masks)"

        # Standard weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # segment params
        self.segment_embeds = None
        if self.use_segment_embeds:
            self.segment_embeds = nn.Embedding(self.segment_size, config.hidden_size)
        
        # prefix params
        self.prefix_embeds = None
        if self.is_prefix_lm or self.is_quasi_lm:
            self.prefix_embeds = nn.Embedding(2, config.hidden_size)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False # found by _xla_set_gradient_checkpointing
        if config._gradient_checkpointing:
            log_print("Gradient checkpointing enabled!")
            self.xla_gradient_checkpointing_enable()

        # Initialize weights and apply final processing
        self.post_init()


    def _get_tokens(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None,
    ) -> torch.Tensor:
        """ Get the tokens that serve as transformer inputs.
         - Convert input_ids
         - Apply segment/prefix embeddings

        Args:
            input_ids (torch.LongTensor): Input token ids [bs, seq_length]
            prefix_length (Optional[torch.LongTensor], optional): Prefix lengths [bs]. Defaults to None.

        Returns:
            torch.Tensor: inputs to transformer [bs, seq_length, hidden_size]
        """
        batch_size, seq_length = input_ids.shape

        # double if this is a quasi LM
        if self.is_quasi_lm:
            input_ids = torch.cat([input_ids, input_ids], dim=1)
            seq_length *= 2

        # get the id tokens
        tokens = self.embed_tokens(input_ids)

        # add segment embeddings
        if self.use_segment_embeds:
            ar = torch.arange(seq_length, device=input_ids.device, dtype=input_ids.dtype)
            segment_ids = ar % self.segment_size
            segment_embs = self.segment_embeds(segment_ids).unsqueeze(0)

            tokens = tokens + segment_embs

        # prefix embedddings
        if self.is_prefix_lm:
            ar = torch.arange(seq_length, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
            prefix_ids = (ar < prefix_length.unsqueeze(-1)).to(input_ids.dtype)
            prefix_embs = self.prefix_embeds(prefix_ids)

            tokens = tokens + prefix_embs

        # quasi embeddings
        if self.is_quasi_lm:
            quasi_ids = torch.zeros_like(input_ids)
            quasi_ids[:, :seq_length//2] = 1
            quasi_embs = self.prefix_embeds(quasi_ids)

            tokens = tokens + quasi_embs

        return tokens
    

    @torch.no_grad()
    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None
    ) -> torch.BoolTensor:
        """ Get the attention mask for the transformer.
         - Handles prefix, quasi, and standard LMs
         - Computations are done on device
         
        Args:
            input_ids (torch.LongTensor): Input token ids [bs, seq_length]
            prefix_length (Optional[torch.LongTensor], optional): Prefix lengths [bs]. Defaults to None.

        Returns:
            torch.BoolTensor: sdpa attention mask [bs|1, 1, seq_length, seq_length]
        """
        batch_size, seq_length = input_ids.shape
    
        # prefix lm is bidirectional for prompt
        if self.is_prefix_lm:

            # get standard mask
            mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

            # apply prefix
            ar = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            p = torch.maximum(ar[:, None], ar[None, :]).unsqueeze(0)
            mask = torch.where(p < prefix_length[:, None, None], torch.zeros_like(mask), mask)
        
        # quasi LM is bidirectional for segments
        elif self.is_quasi_lm:
            if seq_length % self.segment_size != 0:
                raise NotImplementedError("Quasi LM only supports inputs that are tiled by segment size")
            n_segments = seq_length // self.segment_size

            # self attending segments
            nw = torch.ones(1, n_segments, n_segments, dtype=torch.bool, device=input_ids.device)
            nw = torch.triu(nw, diagonal=1)
            nw = torch.repeat_interleave(nw, self.segment_size, dim=1)
            nw = torch.repeat_interleave(nw, self.segment_size, dim=2)

            # segments for cross attention
            sw = torch.ones(1, n_segments, n_segments, dtype=torch.bool, device=input_ids.device)
            sw = torch.triu(sw, diagonal=0)
            sw = torch.repeat_interleave(sw, self.segment_size, dim=1)
            sw = torch.repeat_interleave(sw, self.segment_size, dim=2)

            # empty
            ne = torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)

            # auto regressive within segments
            se = torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
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
            mask = torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
        
        # process for attn version
        if self._attn_implementation == 'eager':
            # eager uses attn bias
            # https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/stablelm/modeling_stablelm.py#L290
            mask = torch.masked_fill(torch.zeros_like(mask).float(), mask, float('-inf'))
        elif self._attn_implementation == 'sdpa':
            # sdpa uses True = NOT masked
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            mask = ~mask
        else:
            # no mask for flash attention
            mask = None

        # final processing
        if mask is not None:

            # cannot broadcast to batch size
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1)

            # head dim
            mask = mask.unsqueeze(1)

        return mask


    @torch.no_grad()
    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """ Get the position ids for the transformer.
         - quasi repeats the sequence

        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length]

        Returns:
            torch.LongTensor: position ids [1, seq_length]
        """
        batch_size, seq_length = input_ids.shape
        
        # standard positions
        pos = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)

        # quasi lm repeats the sequence
        if self.is_quasi_lm:
            pos = torch.cat([pos, pos], dim=1)

        return pos


    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None
    ) -> torch.Tensor:
        """ Forward pass of the LM
         - handles attention masking internally

        TODO: detect padding in prefix of prefix LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            prefix_length (Optional[torch.LongTensor], optional): Prefix lengths [bs]. Defaults to None.

        Returns:
            torch.Tensor: final hidden states [bs, seq_length, hidden_size]
        """
        batch_size, seq_length = input_ids.shape

        # prefix error checking
        if not self.is_prefix_lm:
            assert prefix_length is None, "Prefix length only used for prefix LM"
        else:
            assert prefix_length is not None, "Prefix length required for prefix LM"
            assert isinstance(prefix_length, torch.Tensor), "Prefix length must be a tensor"
            assert tuple(prefix_length.shape) == (batch_size,), "Prefix length must have shape (batch_size,)"

        # get inputs
        hidden_states = self._get_tokens(input_ids, prefix_length)
        mask = self._get_mask(input_ids, prefix_length)
        pos = self._get_position_ids(input_ids)

        # run transformer
        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    mask,
                    pos,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]

        # get decoding states
        if self.is_quasi_lm:
            dec_states = hidden_states[:, -seq_length:]
        else:
            dec_states = hidden_states
        dec_states = self.norm(dec_states)

        # get encoding states
        with torch.no_grad():
            if self.is_quasi_lm:
                enc_states = hidden_states[:, :seq_length]
            else:
                enc_states = hidden_states
            enc_states = self.norm(enc_states).detach()

        return DotDict(
            dec_states=dec_states,
            enc_states=enc_states
        )


class AnnelidLMModel(AnnelidPreTrainedModel):

    def __init__(self, config: AnnelidConfig):
        """ Annelid model with a linear head for language modeling.

        Args:
            config (AnnelidConfig): Annelid configuration
        """
        super().__init__(config)

        # transformer
        self.model = AnnelidModel(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        

    def forward(
        self,
        input_ids: torch.LongTensor,
        prefix_length: Optional[torch.LongTensor]=None,
    ) -> torch.Tensor:
        """ Forward pass of the LM with a linear head.
         - returns log softmaxed logits

        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length]
            prefix_length (Optional[torch.LongTensor], optional): Prefix lengths [bs]. Defaults to None.

        Returns:
            torch.Tensor: log-softmaxed logits [bs, seq_length, vocab_size]
        """

        # final hidden states
        out = self.model(
            input_ids=input_ids,
            prefix_length=prefix_length
        )

        # vocab logits
        logits = self.lm_head(out.dec_states)
        logits = F.log_softmax(logits, dim=-1)

        # encoder logits for eval
        with torch.no_grad():
            enc_logits = self.lm_head(out.enc_states)
            enc_logits = F.log_softmax(enc_logits, dim=-1)

        return DotDict(
            lm_logits=logits,
            enc_logits=enc_logits
        )
