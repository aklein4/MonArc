from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

try:
    from torch_xla.amp import autocast
except:
    pass

from transformers.models.stablelm.modeling_stablelm import (
    StableLmDecoderLayer
)

from models.base import (
    BaseConfig,
    BaseModel,
    BaseTransformer,
    BaseLmModel
)
from utils.data_utils import DotDict
import utils.constants as constants
from utils.logging_utils import log_print


class CortexTransformer(BaseTransformer):

    def __init__(self, config: BaseConfig):
        BaseModel.__init__(self, config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv=None,
        extra_states=None,
    ) -> DotDict:

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # add extras
        if extra_states is not None:
            hidden_states = hidden_states + extra_states

        # previous hidden states for loss
        prev_residual = None

        # run transformer
        for layer_idx, decoder_layer in enumerate(self.layers):

            if layer_idx > 0:
                hidden_states = hidden_states.detach()

            new_hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=kv,
                output_attentions=False,
                use_cache=(kv is not None),
            )[0]

            residual = new_hidden_states - hidden_states.detach()
            hidden_states = new_hidden_states

            if prev_residual is not None:
                
                loss_residual = F.mse_loss(
                    prev_residual,
                    (prev_residual+residual).detach()
                )
                
                with autocast(prev_residual.device, enabled=False):
                    loss_residual.backward()

            prev_residual = residual

        return hidden_states


class CortexLmModel(BaseLmModel):

    def __init__(self, config: BaseConfig):
        BaseModel.__init__(self, config)

        # transformer
        self.model = CortexTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # extras
        self.disable_segment_ids = config.disable_segment_ids

        # Initialize weights and apply final processing
        self.post_init()
