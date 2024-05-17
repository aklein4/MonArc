from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from transformers.models.stablelm.modeling_stablelm import (
    StableLmAttention,
    StableLmMLP,
    apply_rotary_pos_emb,
    repeat_kv
)

from models.base import (
    BaseConfig,
    BaseModel,
    BaseTransformer,
)
from utils.data_utils import DotDict
from utils.logging_utils import log_print


class MonArcConfig(BaseConfig):

    model_type = "monarc"

    def __init__(
        self,
        *args,
        num_head_layers=4,
        control=False,
        head_gradient_checkpointing=False,
        **kwargs
    ):
        self.num_head_layers = num_head_layers
        self.control = control
        self.head_gradient_checkpointing = head_gradient_checkpointing

        super().__init__(*args, **kwargs)


class MonArcCrossAttention(StableLmAttention):

    # copied from StableLmAttention, added memory
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(memory)
        value_states = self.v_proj(memory)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            # Specific to RoPE models with partial rotation
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MonArcHeadLayer(nn.Module):

    # copied from StableLmDecoderLayer, changed attention
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MonArcCrossAttention(config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)


    # copied from StableLmDecoderLayer, changed memory
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Cross Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            memory=memory,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MonArcHeadTransformer(BaseTransformer):

    def __init__(self, config: MonArcConfig):
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MonArcHeadLayer(config, layer_idx) for layer_idx in range(config.num_head_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.head_gradient_checkpointing = config.head_gradient_checkpointing
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        memory: torch.Tensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        kv=None,
    ) -> DotDict:
        batch_size, seq_length = input_ids.shape

        # get inputs
        hidden_states = self._get_tokens(input_ids) + memory
        attention_mask = self._get_mask(input_ids, attention_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training and self.head_gradient_checkpointing:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")
                log_print("Head grad check!")

                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    memory,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    memory=memory,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv,
                    output_attentions=False,
                    use_cache=(kv is not None),
                )[0]

        return self.norm(hidden_states)


class MonArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config, disable_norm=True)
        self.head_model = MonArcHeadTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # fast sampling info
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

        # extras
        self.control = config.control

        # Initialize weights and apply final processing
        self.post_init()

        # init to zero to avoid noise
        self.head_model.embed_tokens.weight.data.zero_()


    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> DotDict:
        """ Forward pass of the LM for training. 
         - creates negative samples
         - returns lm logits and arc predictions
         - 1 in arc predictions is fake, 0 is real, -1 is padding
         
        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length].

        Returns:
            DotDict:
                torch.Tensor: log-softmaxed logits [bs, seq_length, vocab_size]
                torch.Tensor: arc predictions [bs, seq_length-2]
                torch.Tensor: arc targets [bs, seq_length-2]
        """
        batch_size, seq_length = input_ids.shape

        # get lm predictions
        memory = self.model(input_ids)
        # do norm here so it's not applied to the head transformer
        lm_logits = self.lm_head(self.norm(memory))

        # get the true labels
        # the last token is discarded later in the loss
        true_labels = input_ids.clone()
        true_labels[:, :-1] = input_ids[:, 1:]

        # get the fake labels
        fake_labels = input_ids.clone()

        factored_probs = torch.softmax(
            lm_logits.detach().float(), dim=-1
        ).view(-1, self.vocab_factor, self.vocab_chunk)

        outer_probs = factored_probs.sum(dim=-1)
        outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

        ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
        inner_probs = factored_probs[ar, outer_sample]
        inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

        sample = (self.vocab_chunk*outer_sample + inner_sample).view(batch_size, seq_length)
        fake_labels[:, :-1] = sample[:, :-1]

        # get the input tokens for the head
        if self.control:
            true_tokens = torch.zeros_like(true_labels)
            fake_tokens = torch.zeros_like(fake_labels)
        else:
            true_tokens = true_labels
            fake_tokens = fake_labels

        # get the true and fake logits
        true_states = self.head_model(true_tokens, memory)
        # no norm here, head_model handles it
        true_logits = torch.bmm(
            self.lm_head.weight[true_labels.view(-1)].unsqueeze(-2),
            true_states.view(-1, true_states.shape[-1]).unsqueeze(-1)
        )[:, 0]

        fake_states = self.head_model(fake_tokens, memory)
        # # no norm here, head_model handles it
        fake_logits = torch.bmm(
            self.lm_head.weight[fake_labels.view(-1)].unsqueeze(-2),
            fake_states.view(-1, fake_states.shape[-1]).unsqueeze(-1)
        )[:, 0]

        # get arc outputs
        ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
        tmp_lm_logits = lm_logits.view(-1, lm_logits.shape[-1]).detach()
        tmp_true_labels = true_labels.view(-1)
        tmp_fake_labels = fake_labels.view(-1)

        true_arc = true_logits - tmp_lm_logits[ar, tmp_true_labels]
        fake_arc = fake_logits - tmp_lm_logits[ar, tmp_fake_labels]

        # flip sign so higher = lower residual = more likely
        true_arc = -true_arc.view(batch_size, seq_length)
        fake_arc = -fake_arc.view(batch_size, seq_length)

        # final processing
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
