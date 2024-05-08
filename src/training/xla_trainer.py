import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import numpy as np
from tqdm.notebook import tqdm

from training.base_trainer import BaseTrainer

from utils.data_utils import DotDict
import utils.constants as constants


class XLATrainer:

    def __init__(
        self,
        model,
        tokenizer,
        loader,
        lr,
        bs,
        accum_steps,
        num_steps
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.lr = lr
        self.bs = bs
        self.accum_steps = accum_steps
        self.num_steps = num_steps


    def _loss(self, logits, x):
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )


    def train(self):

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

        optimizer = syncfree.AdamW(self.model.parameters(), lr=self.lr)

        tracker = xm.RateTracker()
        for x in self.loader:
            x = torch.split(x, x.shape[0]//self.accum_steps, dim=0)

            optimizer.zero_grad()

            for step in range(self.accum_steps):

                with autocast(constants.XLA_DEVICE()):
                    logits = self.model(x[step])
                    loss = self._loss(logits, x[step])

                loss.backward()
                
            xm.optimizer_step(optimizer)
            
            tracker.add(self.bs)
            print(f"Rate: {tracker.rate()}")
