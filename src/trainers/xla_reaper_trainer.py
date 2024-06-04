import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import (
    loss, ppl, acc, pcorr,
    arc_acc, arc_pcorr, arc_adj,
    reaper_loss, reaper_z
)


class XLAReaperTrainer(BaseXLATrainer):

    def train_step(self, model, x, seg_ids, tokenizer):
        ignore_index = tokenizer.pad_token_id
        
        lm_logits, true_res, fake_res = model.forward(x, segment_ids=seg_ids)

        results = DotDict(
            lm_loss=loss(lm_logits, x, ignore_index),
            lm_ppl=ppl(lm_logits, x, ignore_index),
            lm_acc=acc(lm_logits, x, ignore_index),
            lm_pcorr=pcorr(lm_logits, x, ignore_index),

            reaper_loss=reaper_loss(true_res, fake_res, x, ignore_index),
            reaper_z=reaper_z(fake_res, x, ignore_index),
            arc_acc=arc_acc(true_res, fake_res, x, ignore_index),
            arc_pcorr=arc_pcorr(true_res, fake_res, x, ignore_index),
            arc_adj=arc_adj(true_res, fake_res, x, ignore_index)
        )
        results.loss = (
            results.lm_loss +
            self.w_reap * results.reaper_loss
        )

        # a little extra, as a treat
        results.lm_loss_adj = results.lm_loss.detach() - results.arc_adj

        return results
