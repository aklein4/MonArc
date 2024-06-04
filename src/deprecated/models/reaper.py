from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import (
    BaseConfig,
    BaseModel
)
from models.arc import (
    ArcTransformer
)
from utils.model_utils import EfficientSampler, LogMixture
from utils.data_utils import DotDict
from utils.logging_utils import log_print
import utils.constants as constants


class ReaperConfig(BaseConfig):

    model_type = 'reaper'

    def __init__(
        self,
        *args,
        z_mixture_n: int=2,
        main_mask: int=0,
        main_bias: float=0.0,
        phi_clamp: float=None,
        **kwargs,
    ):
        
        self.z_mixture_n = z_mixture_n
        self.main_mask = main_mask
        self.main_bias = main_bias
        self.phi_clamp = phi_clamp

        super().__init__(*args, **kwargs)


class ReaperLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = ArcTransformer(config, disable_norm=True)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # residual modeling
        self.res_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.forward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.backward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.l_forward_head = nn.Linear(1, config.hidden_size, bias=False)
        self.l_backward_head = nn.Linear(1, config.hidden_size, bias=False)
        self.phi_clamp = config.phi_clamp

        # z prediction
        self.z_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.z_mix = LogMixture(config.hidden_size, config.z_mixture_n, bias=config.main_bias, mask=config.main_mask)

        # extras
        self.sampler = EfficientSampler(self.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        # special initialization
        self.forward_head.weight.data.zero_()
        self.l_forward_head.weight.data.zero_()
        self.l_backward_head.weight.data.zero_()
        self.z_mix.mu.weight.data[0].zero_()
        self.z_mix.sigma.weight.data[0].zero_()
        self.z_mix.pi.weight.data.zero_()


    def _get_lm_logits(
        self,
        hidden_states: torch.Tensor,
    ):
        lm_logits = self.lm_head(self.model.norm(hidden_states))
        return F.log_softmax(lm_logits, dim=-1)


    def _get_residuals(
        self,
        true_states: torch.Tensor,
        fake_states: torch.Tensor,
        input_ids,
        fake_ids,
        lm_logits,
    ):
        batch_size, seq_len = input_ids.shape

        # get state contributions
        true_states = self.res_norm(true_states)
        fake_states = self.res_norm(fake_states)

        forward_embs = self.forward_head(true_states[:, :-1])
        backward_true = self.backward_head(true_states[:, 1:])
        backward_fake = self.backward_head(fake_states[:, 1:])

        # get baseline (logits must be log_softmax!)
        offset_inputs = input_ids.clone()
        offset_inputs[:, :-1] = input_ids[:, 1:]
        offset_fakes = fake_ids.clone()
        offset_fakes[:, :-1] = fake_ids[:, 1:]

        ar = torch.arange(batch_size*seq_len, device=input_ids.device, dtype=input_ids.dtype)
        baseline_true = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_inputs.view(-1)].view(batch_size, seq_len, 1)
        baseline_fake = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_fakes.view(-1)].view(batch_size, seq_len, 1)

        forward_true = forward_embs + self.l_forward_head(baseline_true)[:, :-1]
        forward_fake = forward_embs + self.l_forward_head(baseline_fake)[:, :-1]
        backward_true = backward_true + self.l_backward_head(baseline_true)[:, :-1]
        backward_fake = backward_fake + self.l_backward_head(baseline_fake)[:, :-1]

        # pred[i] = pred for next token, similar to standard LM
        true_res = torch.zeros_like(true_states[:, :, 0])
        fake_res = torch.zeros_like(true_states[:, :, 0])

        # dot product of embs
        true_res[:, :-1] = (forward_true * backward_true).sum(dim=-1) / np.sqrt(self.config.hidden_size)
        fake_res[:, :-1] = (forward_fake * backward_fake).sum(dim=-1) / np.sqrt(self.config.hidden_size)

        if self.phi_clamp is not None:
            true_res = torch.tanh(true_res/self.phi_clamp) * self.phi_clamp
            fake_res = torch.tanh(fake_res/self.phi_clamp) * self.phi_clamp

        return true_res, fake_res


    def _get_z(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = self.z_norm(hidden_states)

        logz_dist = self.z_mix(hidden_states)
        logz = logz_dist.log_mean()

        return logz_dist, logz


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False,
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
        true_res, fake_res = self._get_residuals(
            true_states, fake_states,
            lm_logits=lm_logits, input_ids=input_ids, fake_ids=fake_ids
        )

        # get z prediction
        logz_dist, logz = self._get_z(true_states)

        return (
            lm_logits,
            true_res,
            fake_res,
            logz_dist,
            logz,
            fake_ids
        )


    @torch.no_grad()
    def logp_lm(
        self,
        input_ids,
        segment_ids=None,
    ):

        # get transformer output
        true_states, memory = self.model(
            input_ids=input_ids,
            segment_ids=segment_ids,
        )

        # get lm logits
        lm_logits = self._get_lm_logits(true_states)

        # get z estimate
        logz_dist, logz = self._get_z(true_states)

        return DotDict(
            lm_logits=lm_logits,
            true_states=true_states,
            memory=memory,
            logz_dist=logz_dist,
            logz=logz
        )
    

    @torch.no_grad()
    def residuals(
        self,
        input_ids,
        true_states,
        memory,
        segment_ids=None,
        lm_logits=None
    ):

        fake_states = self.model(
            input_ids,
            memory=memory,
            segment_ids=segment_ids,
        )[0]

        if lm_logits is None:
            lm_logits = self._get_lm_logits(true_states)
        
        true_res, fake_res = self._get_residuals(
            true_states, fake_states,
            lm_logits=lm_logits, input_ids=input_ids, fake_ids=input_ids
        )

        return fake_res
        