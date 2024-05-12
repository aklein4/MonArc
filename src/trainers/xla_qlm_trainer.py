import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLAQLMTrainer(BaseXLATrainer):


    def train_step(self, model, x, tokenizer):
        out = model(x)
        ignore_index = tokenizer.pad_token_id

        results = DotDict(
            lm_loss=loss(out.logits, x, ignore_index),
            lm_ppl=ppl(out.logits, x, ignore_index),
            lm_acc=acc(out.logits, x, ignore_index),
            lm_pcorr=pcorr(out.logits, x, ignore_index),

            enc_loss=loss(out.enc_logits, x, ignore_index),
            enc_ppl=ppl(out.enc_logits, x, ignore_index),
            enc_acc=acc(out.enc_logits, x, ignore_index),
            enc_pcorr=pcorr(out.enc_logits, x, ignore_index)
        )
        results.loss = results.lm_loss

        return results
