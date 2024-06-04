from typing import Optional

import torch

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import os
import numpy as np

import wandb
import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict
from utils.logging_utils import LogSection, log_print, log_master_print


class BaseXLATrainer:

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        debug: Optional[bool] = False
    ):
        """ A trainer to train on TPU devices using PyTorch XLA.

        Args:
            project (str): name of the project to save to
            name (str): name of the run in the project
            config (dict): configuration for the trainer
            debug (bool, optional): Whether to disable saving. Defaults to False.
        """
        self.project = project
        self.name = name
        self.config = config
        self.debug = debug

        save_name = f"{project}_{name}"
        self.save_repo = f"{constants.HF_ID}/{save_name}"

        if constants.XLA_MAIN() and not self.debug:
            with LogSection("Save Locations Creation"):
                hf.create_repo(
                    save_name, private=True, exist_ok=True
                )
                os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)
                wandb.init(
                    project=project,
                    name=name,
                    config=config
                )

        # apply hyperparams
        for k in config:
            setattr(self, k, config[k])

        # init log
        self.log = DotDict()


    def log_step(self):
        if not constants.XLA_MAIN() or self.debug:
            return
        
        # save and clear log
        wandb.log(self.log.to_dict())
        self.log = DotDict()


    @torch.no_grad()
    def save_checkpoint(
        self,
        models,
        step
    ):
        if not constants.XLA_MAIN() or self.debug:
            return
        with LogSection("Saving Checkpoint"):

            api = hf.HfApi()
            tmp_base_path = os.path.join(constants.LOCAL_DATA_PATH, "tmp_checkpoint")
            out_base_path = f"{step:012d}"

            for name, tup in models.items():
                model, on_device = tup

                tmp_path = os.path.join(tmp_base_path, name)
                out_path = os.path.join(out_base_path, name)

                if on_device:
                    os.makedirs(tmp_path, exist_ok=True)
                    xm.save(model.state_dict(), os.path.join(tmp_path, "state_dict.pt"))
                    try:
                        model.config.save_pretrained(tmp_path, push_to_hub=False)
                    except:
                        print(f"Warning: {name} config not saved")
                        pass

                else:
                    model.save_pretrained(tmp_path, push_to_hub=False)

                api.upload_folder(
                    repo_id=self.save_repo,
                    folder_path=tmp_path,
                    path_in_repo=out_path,
                    repo_type="model"
                )
    

    def _get_scheduler(self, optimizer):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        if self.lr_steps is None:
            return warmup_scheduler
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.lr_steps - self.warmup_steps,
            self.end_lr,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
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

        # init training objs
        optimizer = syncfree.AdamW(
            model.parameters(), lr=self.start_lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        lr_scheduler = self._get_scheduler(optimizer)

        # loop
        curr_step = 0
        token_tracker = xm.RateTracker()
        step_tracker = xm.RateTracker()
        for x, seg_ids in loader:
            assert x.shape == seg_ids.shape, f"Input ({x.shape}) and segment ids ({seg_ids.shape}) must have same shape!"

            # prepare x for accum
            n_x = x.shape[0]
            if n_x % self.mini_bs != 0:
                log_print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
            if n_x * constants.NUM_XLA_DEVICES() != self.bs:
                log_print(f"Warning: sample size {n_x} with {constants.NUM_XLA_DEVICES()} devices does not match batch size {self.bs}")
            x_split = torch.split(x, self.mini_bs, dim=0)
            seg_split = torch.split(seg_ids, self.mini_bs, dim=0)

            # accumulate gradients
            results_accum = DotDict()
            for split_idx in range(len(x_split)):
                mini_x = x_split[split_idx]
                mini_seg = seg_split[split_idx]

                # get results from train step
                with autocast(constants.XLA_DEVICE()):
                    results = self.train_step(model, mini_x, mini_seg, tokenizer)

                    # scale results for accumulation
                    for k, v in results.items():
                        results[k] = v / (len(x_split) * constants.NUM_XLA_DEVICES())

                    # save results
                    with torch.no_grad():
                        for k, v in results.items():
                            if k not in results_accum:
                                results_accum[k] = 0.0
                            results_accum[k] = results_accum[k] + v.detach()
                
                results.loss.backward()
                if len(x_split) > 1:
                    xm.mark_step()

            # perform a single optimizer step
            xm.optimizer_step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            
            # update lr
            self.log.lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            # tracking
            token_tracker.add(self.bs * x.shape[1])
            step_tracker.add(1)
            curr_step += 1
            self.log.steps_completed = curr_step

            def _post_step():

                # log
                for k, v in results_accum.items():
                    r = xm.mesh_reduce(f"{k}_reduce", v.item(), np.sum)
                    self.log[k] = r

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
                        },
                        curr_step
                    )
            
            # add closure
            xm.add_step_closure(_post_step)

        self.save_checkpoint(
            {
                'model': (model, True),
                'optimizer': (optimizer, True),
                'tokenizer': (tokenizer, False)
            },
            curr_step
        )
    

    def train_step(
        self,
        model,
        x,
        seg_ids,
        tokenizer
    ):
        """ Get results of a single training step.
         - Must return DotDict of results
         - Results must include 'loss' key

        Args:
            model: model to train
            x: input data [mini_bs, seq_len]
            tokenizer: tokenizer for input data

        Returns:
            DotDict: result tensors containin 'loss' key
        """
        raise NotImplementedError("train_step must be implemented in child class!")