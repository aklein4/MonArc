from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from models.base import (
    BaseConfig, BaseModel
)
from models.arc import ArcTransformer, ArcLmModel
from utils.data_utils import DotDict
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


class ShArcConfig(BaseConfig):

    model_type = 'sharc'

    def __init__(
        self,
        *args,
        sharc_size: int = 1536,
        **kwargs,
    ):
        
        self.sharc_size = sharc_size

        super().__init__(*args, **kwargs)


class ShArcLmModel(ArcLmModel):

    def __init__(self, config: ShArcConfig):
        BaseModel.__init__(self, config)

        # transformer
        self.model = ArcTransformer(config, disable_norm=False)

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


    def _get_lm_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        # norm applied in transformer, apply lm_proj here
        lm_logits = self.lm_head(self.lm_proj(hidden_states))
        return F.log_softmax(lm_logits, dim=-1)


    def _get_arc_outputs(
        self,
        true_states: torch.Tensor,
        fake_states: torch.Tensor,
        input_ids,
        fake_ids,
        lm_logits,
    ):
        batch_size, seq_len = input_ids.shape
        
        # 1. get baseline lm outputs
        # lm_logits should already be log_softmax!
        # lm_logits = F.log_softmax(lm_logits, dim=-1)
        
        # offset inputs index to get y_t[x_{t+1}]
        offset_inputs = input_ids.clone()
        offset_inputs[:, :-1] = input_ids[:, 1:]
        offset_fakes = fake_ids.clone()
        offset_fakes[:, :-1] = fake_ids[:, 1:]

        # get baseline lm outputs l_t[x_{t+1}]
        ar = torch.arange(batch_size*seq_len, device=input_ids.device, dtype=input_ids.dtype)
        baseline_true = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_inputs.view(-1)].view(batch_size, seq_len, 1)
        baseline_fake = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_fakes.view(-1)].view(batch_size, seq_len, 1)

        # get token embeddings e[x_{t+1}]
        emb_true = self.lm_head.weight[offset_inputs]
        emb_fake = self.lm_head.weight[offset_fakes]

        # combine pieces to get sharc mlp inputs
        # everything comes from t, except backward states from t+1
        true_x = torch.cat([true_states[:, :-1], true_states[:, 1:], emb_true[:, :-1], baseline_true[:, :-1]], dim=-1)
        fake_x = torch.cat([true_states[:, :-1], fake_states[:, 1:], emb_fake[:, :-1], baseline_fake[:, :-1]], dim=-1)

        # get mlp outputs
        true_out = self.down_proj(self.act_fn(self.gate_proj(true_x)) * self.up_proj(true_x))[:, :, 0]
        fake_out = self.down_proj(self.act_fn(self.gate_proj(fake_x)) * self.up_proj(fake_x))[:, :, 0]

        # apply offset to align with lm_logits
        true_arc = torch.zeros_like(baseline_true[:, :, 0])
        fake_arc = torch.zeros_like(baseline_fake[:, :, 0])

        true_arc[:, :-1] = true_out
        fake_arc[:, :-1] = fake_out

        return true_arc, fake_arc
