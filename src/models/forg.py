from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.stablelm.modeling_stablelm import (
    StableLmMLP,
    StableLmDecoderLayer,
    ATTENTION_CLASSES
)

from models.base import (
    BaseConfig,
    BaseLmModel,
    BaseTransformer,
)


class ForgDecoderLayer(StableLmDecoderLayer):

    # copied from StableLmDecoderLayer, changed attention
    def __init__(self, config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.use_parallel_residual = False

        self.attn_forg = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp_forg = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):

        residual = hidden_states

        hidden_states_in = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states # + self.attn_forg(hidden_states_in)

        # Fully Connected
        residual = hidden_states
        hidden_states_in = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_in)

        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual # + self.mlp_forg(hidden_states_in)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ForgTransformer(BaseTransformer):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ForgDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class ForgLmModel(BaseLmModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = ForgTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # extras
        self.disable_segment_ids = config.disable_segment_ids

        # Initialize weights and apply final processing
        self.post_init()

        for layer in self.model.layers:
            layer.attn_forg.weight.data.fill_(0)
            layer.mlp_forg.weight.data.fill_(0)
