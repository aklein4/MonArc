import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm.notebook import tqdm

from training.base_trainer import BaseTrainer

from utils.data_utils import DotDict
import utils.constants as constants


class LMTrainer(BaseTrainer):

    _hyperparams = [
        "dtype",
        "lr",
        "bs",
        "accum_steps",
        "num_steps",
        "warmup_steps",
        "save_freq",
        "checkpoint_freq",
    ]

    _metrics = ["loss"]


    def _get_tokens(self, loader, tokenizer, model):
        prompts = loader(self.bs)

        x = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings
        ).input_ids

        if x.shape[1] % model.config.segment_size != 0:
            res = model.config.segment_size - (x.shape[1] % model.config.segment_size)
            x = torch.cat(
                [
                    x,
                    torch.full(
                        (x.shape[0], res),
                        tokenizer.pad_token_id,
                        dtype=x.dtype
                    )
                ],
                dim=1
            )

        return x.to(constants.DEVICE)


    def _loss(self, logits, x, tokenizer):
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            x[:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )


    def train(
        self,
        tokenizer,
        model,
        loader,
    ):

        for p in model.parameters():
            p.requires_grad = True
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        scaler = torch.cuda.amp.GradScaler()

        loader.reset()
        with tqdm(range(self.num_steps), desc="Training") as pbar:
            for step in pbar:

                accum_loss = 0.0
                for accum_step in tqdm(range(self.accum_steps), leave=False):

                    enable_autocast = self.dtype != torch.float32
                    with torch.autocast(
                        device_type=str(constants.DEVICE),
                        dtype=(torch.float16 if not enable_autocast else self.dtype),
                        enabled=enable_autocast
                    ):

                        # handle inputs
                        x = self._get_tokens(loader, tokenizer, model)

                        # get encoding
                        logits = model(x)

                        # get metrics
                        loss = self._loss(logits, x, tokenizer) / self.accum_steps
                    
                    if enable_autocast:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    accum_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad(True)
                lr_scheduler.step()

                # save metrics
                self.log.loss.append(accum_loss)
                pbar.set_postfix({k: v[-1] for k, v in self.log.items()})

                if (step+1) % self.save_freq == 0 or step == self.num_steps-1:
                    self.save()

                if (step+1) % self.checkpoint_freq == 0 or step == self.num_steps-1:
                    self.save_checkpoint(
                        {
                            "model": model
                        }
                    )
                