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
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


class ArcConfig(BaseConfig):

    model_type = "arc"

    def __init__(
        self,
        *args,
        mem_efficient_cross_attn: bool = False,
        **kwargs
    ):
        
        self.mem_efficient_cross_attn = mem_efficient_cross_attn

        super().__init__(*args, **kwargs)


class ArcAttention(StableLmAttention):

    def cross_forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        assert memory.shape == hidden_states.shape, "Memory and hidden states must have the same shape!"
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        mem_key_states = self.k_proj(memory)
        mem_value_states = self.v_proj(memory)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        mem_key_states = mem_key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        mem_value_states = mem_value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos and sin only use dtype of value_states
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            raise ValueError("Cache not supported for ArcAttention!")
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
        mem_key_rot, mem_key_pass = (
            mem_key_states[..., : self.rotary_emb.dim],
            mem_key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        _, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_rot, mem_key_rot = apply_rotary_pos_emb(query_rot, mem_key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        mem_key_states = torch.cat((mem_key_rot, mem_key_pass), dim=-1)

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        mem_key_states = repeat_kv(mem_key_states, self.num_key_value_groups)
        mem_value_states = repeat_kv(mem_value_states, self.num_key_value_groups)

        # apply attention
        attn_weights = torch.matmul(query_states, mem_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        self_attn_weights = (query_states * key_states).sum(dim=-1, keepdim=True) / math.sqrt(self.head_dim)

        # check sizes
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        if self_attn_weights.size() != (bsz, self.num_heads, q_len, 1):
            raise ValueError(
                f"Self Attention weights should be of size {(bsz, self.num_heads, q_len, 1)}, but is"
                f" {self_attn_weights.size()}"
            )

        # apply mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # remove self from attn_weights
        diag_mask = torch.zeros_like(attn_weights[:1, :1])
        diag_mask.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        attn_weights = attn_weights + diag_mask.detach()

        # get attention for all, with self at front
        # upcast attention to fp32
        full_attn = torch.cat([self_attn_weights, attn_weights[..., :-1]], dim=-1)
        attn_out = nn.functional.softmax(full_attn, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_out = self.attention_dropout(attn_out)
        self_attn_weights, attn_weights_tmp = attn_out[..., :1], attn_out[..., 1:]
        attn_weights = torch.zeros_like(attn_weights)
        attn_weights[..., :-1] = attn_weights_tmp 

        # apply attention to values
        attn_output = torch.matmul(attn_weights, mem_value_states)
        attn_output = attn_output + value_states * self_attn_weights

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


    # copied from StableLmAttention, added memory
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        if memory is not None:
            return self.cross_forward(
                hidden_states,
                memory,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )

        return super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )


class ArcDecoderLayer(nn.Module):

    # copied from StableLmDecoderLayer, changed attention
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = ArcAttention(config, layer_idx=layer_idx)
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
        if memory is not None:
            memory = self.input_layernorm(memory)

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


class ArcTransformer(BaseTransformer):

    def __init__(self, config: BaseConfig, disable_norm=False):
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArcDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # more config
        self.disable_norm = disable_norm

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        memory: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv=None,
    ) -> DotDict:

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        mem_out = []
        for layer_idx, decoder_layer in enumerate(self.layers):

            mem_out.append(hidden_states)
            if memory is None:
                mem_in = None
            else:
                mem_in = memory[layer_idx]

            if self.gradient_checkpointing and self.training and layer_idx < self.gradient_checkpointing_layers:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    mem_in,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    memory=mem_in,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv,
                    output_attentions=False,
                    use_cache=(kv is not None),
                )[0]

        # apply final norm
        if not self.disable_norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states, torch.stack(mem_out, dim=0)


class ArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = ArcTransformer(config, disable_norm=True)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # arc modeling
        self.arc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.forward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.backward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.sampler = EfficientSampler(self.vocab_size)

        # compute settings
        self.mem_efficient_cross_attn = config.mem_efficient_cross_attn

        # Initialize weights and apply final processing
        self.post_init()


    def _get_lm_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        lm_logits = self.lm_head(self.model.norm(hidden_states))
        return F.log_softmax(lm_logits, dim=-1)


    def _get_arc_outputs(
        self,
        true_states: torch.Tensor,
        fake_states: torch.Tensor,
    ):
        true_states = self.arc_norm(true_states)
        fake_states = self.arc_norm(fake_states)

        forward_embs = self.forward_head(true_states[:, :-1])
        backward_true = self.backward_head(true_states[:, 1:])
        backward_fake = self.backward_head(fake_states[:, 1:])

        true_arc = torch.zeros_like(true_states[:, :, 0])
        fake_arc = torch.zeros_like(true_states[:, :, 0])

        # pred[i] = pred for next token, similar to standard LM
        true_arc[:, :-1] = (forward_embs * backward_true).sum(dim=-1) / np.sqrt(self.config.hidden_size)
        fake_arc[:, :-1] = (forward_embs * backward_fake).sum(dim=-1) / np.sqrt(self.config.hidden_size)
        
        return true_arc, fake_arc


    def mem_efficient_forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False
    ):

        # get the simple mask and cache
        mask = self.model._get_mask(input_ids, segment_ids=segment_ids)

        # get transformer output
        true_states, memory = self.model(
            input_ids,
            attention_mask=mask,
            cached_mask=True,
        )

        # get lm logits
        lm_logits = self._get_lm_logits(true_states)

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            sample = self.sampler(lm_logits)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # get fake outputs
        fake_states = self.model(
            fake_ids,
            memory=memory,
            attention_mask=mask,
            cached_mask=True,
        )[0]

        # get arc predictions
        true_arc, fake_arc = self._get_arc_outputs(true_states, fake_states)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False
    ):
        if self.mem_efficient_cross_attn:
            return self.mem_efficient_forward(
                input_ids,
                segment_ids,
                debug
            )

        # get the simple mask
        mask = self.model._get_mask(input_ids, segment_ids=segment_ids)
        pos_ids = self.model._get_position_ids(input_ids)

        # logits for sampling
        sample_states = self.model(
            input_ids,
            position_ids=pos_ids,
            attention_mask=mask,
            cached_mask=True,
        )[0].detach()
        sample_logits = self._get_lm_logits(sample_states)

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            sample = self.sampler(sample_logits)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # update inputs for full pass
        input_ids = torch.cat([input_ids, fake_ids], dim=1)
        pos_ids = torch.cat([pos_ids, pos_ids], dim=1)

        nw = mask.clone()
        ne = torch.full_like(mask, float("-inf"))
        sw = mask.clone()
        sw.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        se = torch.full_like(mask, float("-inf"))
        se.diagonal(dim1=-2, dim2=-1).fill_(0)
        mask = torch.cat(
            [
                torch.cat([nw, ne], dim=-1),
                torch.cat([sw, se], dim=-1)
            ],
            dim=-2
        )

        # get fake outputs
        hidden_states = self.model(
            input_ids,
            position_ids=pos_ids,
            attention_mask=mask,
            cached_mask=True,
        )[0]
        true_states, fake_states = hidden_states.chunk(2, dim=-2)

        # get lm logits
        lm_logits = self._get_lm_logits(true_states)

        # get arc predictions
        true_arc, fake_arc = self._get_arc_outputs(true_states, fake_states)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
    