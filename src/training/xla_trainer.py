import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_xla.amp import autocast
import torch_xla.core.xla_model as xm
from torch_xla.amp import syncfree

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
        num_steps
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.lr = lr
        self.bs = bs
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

        optimizer = syncfree.AdamW(self.model.parameters(), lr=self.lr*xm.xrt_world_size())
        
        tracker = xm.RateTracker()
        x = torch.zeros((self.bs, 1024), dtype=torch.long).to(constants.XLA_DEVICE)
        for _ in range(self.num_steps):

            optimizer.zero_grad()

            # with autocast():
            logits = self.model(x)
            loss = self._loss(logits, x)

            loss.backward()
            xm.optimizer_step(optimizer, barrier=False)

            tracker.add(self.bs)
            