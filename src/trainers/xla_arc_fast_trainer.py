import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLAArcFastTrainer(BaseXLATrainer):


    def train_step(self, model, x, tokenizer):
        ignore_index = tokenizer.pad_token_id

        negative_samples = model.sample_negatives(x, tokenizer.pad_token_id)
        out = model.forward_from_sample(x, negative_samples, tokenizer.pad_token_id)

        results = DotDict(
            lm_loss=loss(out.lm_logits, x, ignore_index),
            lm_ppl=ppl(out.lm_logits, x, ignore_index),
            lm_acc=acc(out.lm_logits, x, ignore_index),
            lm_pcorr=pcorr(out.lm_logits, x, ignore_index),

            arc_loss=loss(out.arc_preds, out.arc_targets, -1),
            arc_ppl=ppl(out.arc_preds, out.arc_targets, -1),
            arc_acc=acc(out.arc_preds, out.arc_targets, -1),
            arc_pcorr=pcorr(out.arc_preds, out.arc_targets, -1)
        )
        results.loss = results.lm_loss + self.w_arc * results.arc_loss

        return results
