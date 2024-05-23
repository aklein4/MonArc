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
        self.forward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.backward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.sampler = EfficientSampler(self.vocab_size)

        # reparameterization
        self.baseline_forward_head = nn.Linear(1, config.hidden_size, bias=False)
        self.baseline_backward_head = nn.Linear(1, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # head bias is not init to zero
        self.forward_head.bias.data.normal_(mean=0.0, std=config.initializer_range)
        self.backward_head.bias.data.normal_(mean=0.0, std=config.initializer_range)


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
        lm_logits = F.log_softmax(lm_logits, dim=-1)
        
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

        forward_embs = self.forward_head(true_states[:, :-1])
        backward_true = self.backward_head(true_states[:, 1:])
        backward_fake = self.backward_head(fake_states[:, 1:])

        # 3. project baseline into embedding space
        forward_embs = forward_embs + self.baseline_forward_head(baseline_true[:, :-1])
        backward_true = backward_true + self.baseline_backward_head(baseline_true[:, :-1])
        backward_fake = backward_fake + self.baseline_backward_head(baseline_fake[:, :-1])

        # 4. multiply embeddings to get arc outputs
        true_arc = torch.zeros_like(true_states[:, :, 0])
        fake_arc = torch.zeros_like(true_states[:, :, 0])

        # pred[i] = pred for next token, similar to standard LM
        true_arc[:, :-1] = (forward_embs * backward_true).sum(dim=-1) / np.sqrt(self.config.hidden_size)
        fake_arc[:, :-1] = (forward_embs * backward_fake).sum(dim=-1) / np.sqrt(self.config.hidden_size)

        return true_arc, fake_arc
