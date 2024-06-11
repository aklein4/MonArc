from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from transformers.models.stablelm.modeling_stablelm import (
    StableLmAttention,
    StableLmMLP,
    StableLmDecoderLayer
)

from models.base import (
    BaseConfig,
    BaseModel,
    BaseTransformer,
)
from utils.data_utils import DotDict
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


class HydeConfig(BaseConfig):

    model_type = 'hyde'

    def __init__(
        self,
        *args,
        attn_size: int=768,
        output_factor: int=None,
        input_factor: int=None,
        **kwargs
    ):
        
        self.attn_size = attn_size
        self.output_factor = output_factor
        self.input_factor = input_factor

        super().__init__(*args, **kwargs)


class HydeAttention(StableLmAttention):

    def __init__(self, config: HydeConfig, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.attn_size
        self.external_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.external_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(self.external_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(self.external_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.external_size, bias=False)

        self.qk_layernorm = False

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self._init_rope()


class HydeDecoderLayer(StableLmDecoderLayer):

    # copied from StableLmDecoderLayer, changed attention
    def __init__(self, config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = HydeAttention(config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)


class HydeTransformer(BaseTransformer):

    def __init__(self, config: BaseConfig):
        BaseModel.__init__(self, config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        if config.input_factor is not None:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.input_factor, self.padding_idx)
            self.embed_proj = nn.Linear(config.input_factor, config.hidden_size, bias=False)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.embed_proj = nn.Identity()
        self.layers = nn.ModuleList(
            [HydeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def _get_tokens(
        self,
        input_ids: torch.LongTensor
    ) -> torch.Tensor:
        tokens = self.embed_tokens(input_ids)
        return self.embed_proj(tokens)


class HydeLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = HydeTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        if config.output_factor is not None:
            self.lm_factorizer = nn.Linear(config.hidden_size, config.output_factor, bias=False)
            self.lm_head = nn.Linear(config.output_factor, config.vocab_size, bias=False)
        else:
            self.lm_factorizer = nn.Identity()
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        kv=None,
    ) -> DotDict:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.
            kv (Optional[Cache], optional): Key-Value cache. Defaults to None.
        
        Returns:
            DotDict:
                lm_logits: log-softmaxed token probs [bs, seq_length, vocab_size]
        """

        # get lm predictions
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            kv=kv
        )

        lm_logits = self.lm_head(self.lm_factorizer(out))
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits