import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast

import numpy as np
from tqdm.notebook import tqdm

from training.base_xla_trainer import BaseXLATrainer

from utils.data_utils import DotDict
import utils.constants as constants


class XLATrainer(BaseXLATrainer):

    _metrics = ["loss"]


    def _loss(self, logits, x, tokenizer):
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            x[:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )


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
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        # loop
        tracker = xm.RateTracker()
        for x in loader:

            # prepare x for accum
            n_x = x.shape[0]
            if n_x % self.mini_bs != 0:
                print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
            x = torch.split(x, self.mini_bs, dim=0)

            # accumulate gradients
            loss_accum = 0.0
            for mini_x in x:

                with autocast(constants.XLA_DEVICE()):
                    logits = model(mini_x)
                    loss = self._loss(logits, mini_x, tokenizer)

                # scale loss to the sample size
                loss = loss / len(x)
                loss = loss / constants.NUM_XLA_DEVICES()

                # mark step to save gradients
                loss.backward()
                xm.mark_step()

                # save loss
                loss_accum += loss.item()

            # perform a single optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            
            # log
            log_loss = xm.mesh_reduce("loss_reduce", loss_accum, np.sum)
            self.log["loss"].append(log_loss)
            tracker.add(self.bs)

            xm.master_print(f"Step {len(self.log['loss'])}: Loss = {log_loss:.4f} | {tracker.rate():.2f} samples/s")
                
            if len(self.log["loss"]) % self.save_interval == 0:
                self.save()
            if len(self.log["loss"]) % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    {
                        'model': model,
                        'tokenizer': tokenizer
                    }
                )
        
        self.save()
        self.save_checkpoint()
        