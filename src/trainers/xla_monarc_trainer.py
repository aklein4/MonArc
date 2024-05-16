import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLAMonArcTrainer(BaseXLATrainer):


    def train_step(self, model, x, tokenizer):
        lm_logits, arc_preds, arc_targets = model.forward(x)
        ignore_index = tokenizer.pad_token_id

        results = DotDict(
            lm_loss=loss(lm_logits, x, ignore_index),
            lm_ppl=ppl(lm_logits, x, ignore_index),
            lm_acc=acc(lm_logits, x, ignore_index),
            lm_pcorr=pcorr(lm_logits, x, ignore_index),

            arc_loss=loss(arc_preds, arc_targets, -1),
            arc_ppl=ppl(arc_preds, arc_targets, -1),
            arc_acc=acc(arc_preds, arc_targets, -1),
            arc_pcorr=pcorr(arc_preds, arc_targets, -1)
        )
        results.loss = results.lm_loss + self.w_arc * results.arc_loss

        return results
