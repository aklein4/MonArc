import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import numpy as np

from training.base_xla_trainer import BaseXLATrainer
from utils.logging_utils import log_master_print
import utils.constants as constants
from utils.data_utils import DotDict


class XLATrainer(BaseXLATrainer):

    def _loss(self, logits, x, tokenizer):
        x, logits = x[:, 1:], logits[:, :-1]

        return F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            x.contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )
    

    def _acc(self, logits, x, tokenizer):
        x, logits = x[:, 1:], logits[:, :-1]
        mask = x != tokenizer.pad_token_id

        corr = torch.logical_and(
            logits.argmax(-1) == x,
            mask
        ).float().sum()
        return corr / (mask).float().sum()


    def _pcorr(self, logits, x, tokenizer):
        x = x[:, 1:].contiguous().view(-1)
        logits = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
        mask = x != tokenizer.pad_token_id

        logp = -F.cross_entropy(
            logits, x,
            reduction='none'
        )
        p = torch.exp(logp)

        p = torch.masked_fill(p, ~mask, 0.0)
        return p.sum() / (mask).float().sum()


    def all_results(self, logits, x, tokenizer):
        return DotDict(
            loss=self._loss(logits, x, tokenizer),
            acc=self._acc(logits, x, tokenizer),
            pcorr=self._pcorr(logits, x, tokenizer)
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
        optimizer = syncfree.AdamW(
            model.parameters(), lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )

        # test
        self.save_checkpoint(
            {
                'model': (model, True),
                'optimizer': (optimizer, True),
                'tokenizer': (tokenizer, False)
            }
        )

        # loop
        curr_step = 0
        token_tracker = xm.RateTracker()
        step_tracker = xm.RateTracker()
        for x in loader:

            # prepare x for accum
            n_x = x.shape[0]
            if n_x % self.mini_bs != 0:
                print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
            if n_x * constants.NUM_XLA_DEVICES() != self.bs:
                print(f"Warning: sample size {n_x} with {constants.NUM_XLA_DEVICES()} devices does not match batch size {self.bs}")
            x_split = torch.split(x, self.mini_bs, dim=0)

            # accumulate gradients
            results_accum = DotDict().from_dict({k: 0.0 for k in self._metrics})
            for mini_x in x_split:

                with autocast(constants.XLA_DEVICE()):
                    logits = model(mini_x)
                    results = self.all_results(logits, mini_x, tokenizer)

                    # scale for additive reduction
                    for k in self._metrics:
                        results[k] = results[k] / len(x_split)
                        results[k] = results[k] / constants.NUM_XLA_DEVICES()

                # mark step to save gradients
                results.loss.backward()
                xm.mark_step()

                # save results
                for k in self._metrics:
                    results_accum[k] = results_accum[k] + results[k].detach()

            # perform a single optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # log
            for k, v in results_accum.items():
                r = xm.mesh_reduce(f"{k}_reduce", v.item(), np.sum)
                self.log[k] = r
            token_tracker.add(self.bs * x.shape[1])
            step_tracker.add(1)
            curr_step += 1

            # print update
            msg = [
                f"Step {curr_step}",
                f"Loss = {self.log.loss:.4f}",
                f"Acc = {self.log.acc:.3f}",
                f"PCorr = {self.log.pcorr:.3f}",
                f"{step_tracker.rate():.2f} steps/s",
                f"{round(3600*token_tracker.rate()):_} tok/h"
            ]
            log_master_print("{: >15}{: >20}{: >20}{: >20}{: >20}{: >23}".format(*msg))
            
            # save
            self.log_step()
            if curr_step % self.checkpoint_interval == 0:
                self.save_checkpoint(
                    {
                        'model': (model, True),
                        'optimizer': (optimizer, True),
                        'tokenizer': (tokenizer, False)
                    }
                )
        
        self.save()
        self.save_checkpoint(
            {
                'model': (model, True),
                'optimizer': (optimizer, True),
                'tokenizer': (tokenizer, False)
            }
        )
