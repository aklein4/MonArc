from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from models.base import BaseModel
from models.arc import (
    ArcTransformer, 
    ArcDecoderLayer
)
from models.sharc import (
    ShArcConfig,
    ShArcLmModel
)
from utils.data_utils import DotDict
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


# changed vocab size of embed_tokens to 2*config.vocab_size
class RemArcTransformer(ArcTransformer):

    def __init__(self, config, disable_norm=False):
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(2*config.vocab_size, config.hidden_size, self.padding_idx)
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


class RemArcLmModel(ShArcLmModel):

    def __init__(self, config: ShArcConfig):
        BaseModel.__init__(self, config)

        # transformer
        self.model = RemArcTransformer(config, disable_norm=False)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        # we project into the lm head to seperate lm and arc subspaces
        self.lm_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # sharc mlp based on StableLmMLP
        # takes forward states, backwards states, token emb, and baseline lm output
        self.gate_proj = nn.Linear(1+(3*config.hidden_size), config.sharc_size, bias=False)
        self.up_proj = nn.Linear(1+(3*config.hidden_size), config.sharc_size, bias=False)
        self.down_proj = nn.Linear(config.sharc_size, 1, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # helpers
        self.sampler = EfficientSampler(self.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        # init weights for better stability
        self.lm_proj.weight.data.copy_(torch.eye(config.hidden_size))
        self.down_proj.weight.data.zero_()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False,
        return_dict=False
    ):

        # get the simple mask and cache
        mask = self.model._get_mask(input_ids, segment_ids=segment_ids)

        # get transformer output
        lm_states, memory = self.model(
            input_ids,
            attention_mask=mask,
            cached_mask=True,
        )

        # get lm logits
        lm_logits = self._get_lm_logits(lm_states)

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            sample = self.sampler(lm_logits)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # get fake outputs
        true_states, fake_states = self.model(
            torch.cat([input_ids, fake_ids], dim=0) + self.vocab_size,
            memory=torch.cat([memory, memory], dim=1),
            attention_mask=torch.cat([mask, mask], dim=0),
            cached_mask=True,
        )[0].chunk(2, dim=0)

        # get arc predictions
        true_arc, fake_arc = self._get_arc_outputs(
            true_states, fake_states,
            lm_logits=lm_logits, input_ids=input_ids, fake_ids=fake_ids
        )

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
    