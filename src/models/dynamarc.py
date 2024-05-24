from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from models.base import (
    BaseConfig, BaseModel
)
from models.arc import ArcTransformer, ArcLmModel
from utils.data_utils import DotDict
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


class DynamArcLmModel(ArcLmModel):

    def __init__(self, config: BaseConfig):
        BaseModel.__init__(self, config)

        # transformer
        self.model = ArcTransformer(config, disable_norm=True)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # arc modeling
        self.arc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # embedding projections (no bias)
        self.forward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.backward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # linear heads on top of outputs (forward_bias contains global bias)
        self.forward_bias = nn.Linear(config.hidden_size, 1, bias=True)
        self.backward_bias = nn.Linear(config.hidden_size, 1, bias=False)

        self.sampler = EfficientSampler(self.vocab_size)

        # reparameterization (baseline_forward_head contains baseline bias)
        self.baseline_forward_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.baseline_backward_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


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
        
        offset_inputs = input_ids.clone()
        offset_inputs[:, :-1] = input_ids[:, 1:]
        offset_fakes = fake_ids.clone()
        offset_fakes[:, :-1] = fake_ids[:, 1:]

        ar = torch.arange(batch_size*seq_len, device=input_ids.device, dtype=input_ids.dtype)
        baseline_true = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_inputs.view(-1)].view(batch_size, seq_len, 1)
        baseline_fake = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_fakes.view(-1)].view(batch_size, seq_len, 1)

        # 2. get arc embeddings
        true_states = self.arc_norm(true_states)
        fake_states = self.arc_norm(fake_states)

        # get embedddings
        forward_embs = self.forward_head(true_states[:, :-1])
        backward_true_embs = self.backward_head(true_states[:, 1:])
        backward_fake_embs = self.backward_head(fake_states[:, 1:])

        # get biases
        forward_bias = self.forward_bias(true_states[:, :-1])[:, :, 0]
        backward_true_bias = self.backward_bias(true_states[:, 1:])[:, :, 0]
        backward_fake_bias = self.backward_bias(fake_states[:, 1:])[:, :, 0]

        # get baseline scales
        forward_baseline = self.baseline_forward_head(true_states[:, :-1])[:, :, 0]
        backward_true_baseline = self.baseline_backward_head(true_states[:, 1:])[:, :, 0]
        backward_fake_baseline = self.baseline_backward_head(fake_states[:, 1:])[:, : 0]

        # 4. combine embeddings to get arc outputs
        true_arc = torch.zeros_like(true_states[:, :, 0])
        fake_arc = torch.zeros_like(true_states[:, :, 0])

        # dot product of embs
        true_arc[:, :-1] = (forward_embs * backward_true_embs).sum(dim=-1) / np.sqrt(self.config.hidden_size)
        fake_arc[:, :-1] = (forward_embs * backward_fake_embs).sum(dim=-1) / np.sqrt(self.config.hidden_size)

        # biases
        true_arc[:, :-1] = true_arc[:, :-1] + forward_bias + backward_true_bias
        fake_arc[:, :-1] = fake_arc[:, :-1] + forward_bias + backward_fake_bias

        # baseline
        log_print(true_arc.shape, forward_baseline.shape, backward_true_baseline.shape, baseline_true.shape)
        true_arc[:, :-1] = true_arc[:, :-1] + (forward_baseline + backward_true_baseline)*baseline_true[:, :-1]
        fake_arc[:, :-1] = fake_arc[:, :-1] + (forward_baseline + backward_fake_baseline)*baseline_fake[:, :-1]

        return true_arc, fake_arc
