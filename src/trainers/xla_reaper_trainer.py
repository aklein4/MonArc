import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import (
    loss, ppl, acc, pcorr,
    reaper_phi_loss, reaper_z_loss, reaper_adj,
    reaper_sample_abs, reaper_mu_abs, reaper_sigma,
    reaper_check, reaper_sample, reaper_mu
)


class XLAReaperTrainer(BaseXLATrainer):

    def train_step(self, model, x, seg_ids, tokenizer):
        ignore_index = tokenizer.pad_token_id
        
        lm_logits, true_res, fake_res, mu, sigma, logz = model.forward(x, segment_ids=seg_ids)

        results = DotDict(
            lm_loss=loss(lm_logits, x, ignore_index),
            lm_ppl=ppl(lm_logits, x, ignore_index),
            lm_acc=acc(lm_logits, x, ignore_index),
            lm_pcorr=pcorr(lm_logits, x, ignore_index),

            reaper_phi_loss=reaper_phi_loss(true_res, fake_res, logz, x, ignore_index),
            reaper_z_loss=reaper_z_loss(true_res, fake_res, mu, sigma, x, ignore_index),
            arc_adj=reaper_adj(true_res, fake_res, logz, x, ignore_index),
            reaper_sample_abs=reaper_sample_abs(fake_res, x, ignore_index),
            reaper_mu_abs=reaper_mu_abs(mu, x, ignore_index),
            reaper_sigma=reaper_sigma(sigma, x, ignore_index),
            reaper_check=reaper_check(true_res, fake_res, logz, x, ignore_index),
            reaper_sample=reaper_sample(fake_res, x, ignore_index),
            reaper_mu=reaper_mu(mu, x, ignore_index),
        )
        results.loss = (
            results.lm_loss +
            self.w_phi * results.reaper_phi_loss +
            self.w_z * results.reaper_z_loss
        )

        # a little extra, as a treat
        results.lm_loss_adj = results.lm_loss.detach() - results.arc_adj

        return results
