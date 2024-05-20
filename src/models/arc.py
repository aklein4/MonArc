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
from transformers.cache_utils import DynamicCache

from models.base import (
    BaseConfig,
    BaseModel,
    BaseTransformer,
)
from utils.data_utils import DotDict
from utils.logging_utils import log_print



class ArcConfig(BaseConfig):

    model_type = "arc"

    def __init__(
        self,
        *args,
        arc_divisor: int = 1,
        **kwargs
    ):
        self.arc_divisor = arc_divisor

        super().__init__(*args, **kwargs)


class ArcAttention(StableLmAttention):


    def arc_forward(
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


    def fast_forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # apply memory
        hidden_states = torch.cat([memory, hidden_states], dim=1)
        position_ids = torch.cat([position_ids, position_ids], dim=1)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            raise NotImplementedError("Cache not supported for ArcAttention!")
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

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # remove memory from query_states
        query_states = query_states[:, :, memory.shape[1]:]

        # apply attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # check sizes
        if attn_weights.size() != (bsz, self.num_heads, q_len//2, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply mask with memory
        left_mask = torch.zeros_like(attention_mask)
        left_mask.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        left_mask = left_mask.detach() + attention_mask

        right_mask = torch.full_like(attention_mask, float("-inf"))
        right_mask.diagonal(dim1=-2, dim2=-1).fill_(0)
        right_mask = right_mask.detach() + attention_mask

        attention_mask = torch.cat([left_mask, right_mask], dim=-1)

        attn_weights = attn_weights + attention_mask

        # get attention for all, with self at front
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights) 

        # apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len//2, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len//2, self.hidden_size)

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
            return self.fast_forward(
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

    def __init__(self, config: BaseConfig):
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

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor]=None,
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

        return self.norm(hidden_states), torch.stack(mem_out, dim=0)


class ArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # arc modeling
        self.arc_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.arc_divisor = config.arc_divisor

        # fast sampling info
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False
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

        # get the simple mask and cache
        mask = self.model._get_mask(input_ids, segment_ids=segment_ids)
        kv = DynamicCache()

        # get transformer output
        true_states = self.model(
            input_ids,
            segment_ids=segment_ids,
            attention_mask=mask,
            cached_mask=True,
            kv=kv
        )

        # get lm logits
        lm_logits = self.lm_head(true_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        # # apply the arc divisor
        # assert batch_size % self.arc_divisor == 0, f"Batch size {batch_size} must be divisible by {self.arc_divisor}!"
        # batch_size = batch_size // self.arc_divisor

        # input_ids = input_ids[:batch_size]
        # segment_ids = segment_ids[:batch_size]
        # memory = memory[:, :batch_size]
        # sample_logits = lm_logits[:batch_size]

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            factored_probs = torch.softmax(
                lm_logits.detach().float(), dim=-1
            ).view(-1, self.vocab_factor, self.vocab_chunk)

            outer_probs = factored_probs.sum(dim=-1)
            outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

            ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
            inner_probs = factored_probs[ar, outer_sample]
            inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

            sample = (self.vocab_chunk*outer_sample + inner_sample).view(batch_size, seq_length)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # get the new mask
        left_mask = torch.zeros_like(mask[:1])
        left_mask.diagonal(dim1=-2, dim2=-1).fill_(float("-inf"))
        left_mask = left_mask.detach() + mask

        right_mask = torch.full_like(mask[:1], float("-inf"))
        right_mask.diagonal(dim1=-2, dim2=-1).fill_(0)
        right_mask = right_mask.detach() + mask

        mask = torch.cat([left_mask, right_mask], dim=-1)

        # get fake outputs
        fake_states = self.model(
            fake_ids,
            segment_ids=segment_ids,
            attention_mask=mask,
            cached_mask=True,
            kv=kv
        )

        # get arc predictions
        true_arc = self.arc_head(true_states)[:, :, 0]
        fake_arc = self.arc_head(fake_states)[:, :, 0]

        # shift predictions to account for later shift in loss
        true_arc[:, :-1] = true_arc[:, 1:]
        fake_arc[:, :-1] = fake_arc[:, 1:]

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
