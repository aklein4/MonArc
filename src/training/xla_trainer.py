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
    

    @torch.no_grad()
    def _ppl(self, logits, x, tokenizer):
        x = x[:, 1:]
        logits = logits[:, :-1]
        mask = x != tokenizer.pad_token_id

        logp = -F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            x.contiguous().view(-1),
            reduction='none'
        ).reshape(x.shape)

        logp = torch.masked_fill(logp, ~mask, 0.0)
        logp_seq = logp.sum(-1) / (mask).float().sum(-1)

        return torch.exp(-logp_seq).mean()


    @torch.no_grad()
    def _acc(self, logits, x, tokenizer):
        x, logits = x[:, 1:], logits[:, :-1]
        mask = x != tokenizer.pad_token_id

        corr = torch.logical_and(
            logits.argmax(-1) == x,
            mask
        ).float().sum()
        return corr / (mask).float().sum()


    @torch.no_grad()
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
            ppl=self._ppl(logits, x, tokenizer),
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
            model.parameters(), lr=self.start_lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.lr_steps - self.warmup_steps,
            self.end_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
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
            results_accum = DotDict()
            for mini_x in x_split:

                with autocast(constants.XLA_DEVICE()):
                    out = model(mini_x)

                    results = self.all_results(out.logits, mini_x, tokenizer)
                    for k in results.keys(): # scale for additive reduction
                        results[k] = results[k] / len(x_split)
                        results[k] = results[k] / constants.NUM_XLA_DEVICES()
                    
                    with torch.no_grad():
                        enc_results = self.all_results(out.enc_logits, mini_x, tokenizer)
                        for k in enc_results.keys(): # scale for additive reduction
                            enc_results[k] = enc_results[k] / len(x_split)
                            enc_results[k] = enc_results[k] / constants.NUM_XLA_DEVICES()

                # mark step to save gradients
                results.loss.backward()
                xm.mark_step()

                # save results
                with torch.no_grad():
                    for k, v in results.items():
                        if k in results_accum:
                            results_accum[k] = results_accum[k] + v.detach()
                        else:
                            results_accum[k] = v.detach()
                    for k, v in enc_results.items():
                        enck = f"enc_{k}"
                        if enck in results_accum:
                            results_accum[enck] = results_accum[enck] + v.detach()
                        else:
                            results_accum[enck] = v.detach()
                
            # perform a single optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            
            # log
            for k, v in results_accum.items():
                r = xm.mesh_reduce(f"{k}_reduce", v.item(), np.sum)
                self.log[k] = r

            # update lr
            self.log.lr = lr_scheduler.get_last_lr()[0]
            if not isinstance(self.log.lr, float):
                self.log.lr = 0.0
            lr_scheduler.step()

            # tracking
            token_tracker.add(self.bs * x.shape[1])
            step_tracker.add(1)
            curr_step += 1

            # print update
            msg = [
                f"Step {curr_step}",
                f"LR = {self.log.lr:.2e}",
                f"Loss = {self.log.loss:.4f}",
                f"{step_tracker.rate():.2f} steps/s",
                f"{round(3600*token_tracker.rate()):_} tok/h"
            ]
            log_master_print("{: >15}{: >20}{: >20}{: >20}{: >23}".format(*msg))
            
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
        
        self.save_checkpoint(
            {
                'model': (model, True),
                'optimizer': (optimizer, True),
                'tokenizer': (tokenizer, False)
            }
        )
