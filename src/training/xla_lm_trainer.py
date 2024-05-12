import torch
import torch.nn as nn
import torch.nn.functional as F

from training.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import loss, ppl, acc, pcorr


class XLALMTrainer(BaseXLATrainer):


    def train_step(self, model, x, tokenizer):
        out = model(x)

        return DotDict(
            loss=loss(out.logits, x, tokenizer),
            ppl=ppl(out.logits, x, tokenizer),
            acc=acc(out.logits, x, tokenizer),
            pcorr=pcorr(out.logits, x, tokenizer),

            enc_loss=loss(out.enc_logits, x, tokenizer),
            enc_ppl=ppl(out.enc_logits, x, tokenizer),
            enc_acc=acc(out.enc_logits, x, tokenizer),
            enc_pcorr=pcorr(out.enc_logits, x, tokenizer)
        )
