import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import numpy as np
from tqdm.notebook import tqdm

from training.base_xla_trainer import BaseXLATrainer

from utils.data_utils import DotDict
import utils.constants as constants


class XLATrainer(BaseXLATrainer):

    _metrics = ["loss", "acc", "pcorr"]


    def _loss(self, logits, x, tokenizer):
        x, logits = x[:, 1:], logits[:, :-1]

        return F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            x.contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )
    

    def _acc(self, logits, x, tokenizer):
        x, logits = x[:, 1:], logits[:, :-1]

        corr = (
            logits.argmax(-1) == x and 
            x != tokenizer.pad_token_id
        ).float().sum()
        return corr / (x != tokenizer.pad_token_id).float().sum()


    def _pcorr(self, logits, x, tokenizer):
        x = x[:, 1:].contiguous().view(-1)
        logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])

        logp = F.cross_entropy(
            logits, x
            reduce='none'
        )
        p = torch.exp(-logp)

        p[x == tokenizer.pad_token_id] = 0.0
        return p / (x != tokenizer.pad_token_id).float().sum()


    def train(
        self,
        model,
        tokenizer,
        loader
    ):

        # init model
        for p in model.parameters():
            p.requires_grad = True
        model.train()

        # get optimizer
        optimizer = syncfree.AdamW(model.parameters(), lr=self.lr)

        # loop
        token_tracker = xm.RateTracker()
        step_tracker = xm.RateTracker()
        for x in loader:

            # prepare x for accum
            n_x = x.shape[0]
            if n_x % self.mini_bs != 0:
                print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
            x_split = torch.split(x, self.mini_bs, dim=0)

            # accumulate gradients
            loss_accum = 0.0
            for mini_x in x_split:

                with autocast(constants.XLA_DEVICE()):
                    logits = model(mini_x)
                    loss = self._loss(logits, mini_x, tokenizer)

                # scale loss to the sample size
                loss = loss / len(x_split)
                loss = loss / constants.NUM_XLA_DEVICES()

                # mark step to save gradients
                loss.backward()
                xm.mark_step()

                # save loss
                loss_accum = loss_accum + loss.detach()

            # perform a single optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            
            # log
            log_loss = xm.mesh_reduce("loss_reduce", loss_accum.item(), np.sum)
            self.log["loss"].append(log_loss)
            token_tracker.add(self.bs * x.shape[1])
            step_tracker.add(1)

            # print update
            msg = [f"Step {len(self.log['loss'])}", f"Loss = {log_loss:.4f}", f"{step_tracker.rate():.2f} steps/s", f"{round(3600*token_tracker.rate()):_} tokens/h"]
            xm.master_print("{: >15} {: >20} {: >20} {: >25}".format(*msg))
            
            # save
            if len(self.log["loss"]) % self.save_interval == 0:
                self.save()
            if len(self.log["loss"]) % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    {
                        'model': (model, True),
                        'tokenizer': (tokenizer, False)
                    }
                )
        
        self.save()
        self.save_checkpoint(
            {
                'model': (model, True),
                'tokenizer': (tokenizer, False)
            }
        )
