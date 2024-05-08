import torch

import torch_xla.core.xla_model as xm

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict


class BaseXLATrainer:

    _hyper_file = os.path.join(constants.LOCAL_DATA_PATH, "hyperparams.yml")
    _log_file = os.path.join(constants.LOCAL_DATA_PATH, "log.csv")
    _progress_file = os.path.join(constants.LOCAL_DATA_PATH, "progress.png")

    _metrics = []

    def __init__(
        self,
        save_name,
        config
    ):
        self.save_name = save_name
        self.save_repo = f"{constants.HF_ID}/{save_name}"
        self.config = config

        # TODO: some kind of lock?
        if constants.XLA_MAIN():
            hf.create_repo(
                save_name, private=True, exist_ok=True
            )
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)
            print("Created repo and local data path.")

        # apply hyperparams
        for k in config:
            setattr(self, k, config[k])

        self.log = DotDict()
        for m in self._metrics:
            self.log[m] = []


    @torch.no_grad()
    def save(self):
        if not constants.XLA_MAIN():
            return
        print("Saving...")

        # save hyperparams as csv
        with open(self._hyper_file, 'w') as outfile:
            yaml.dump(
                self.config,
                outfile,
                default_flow_style=False
            )

        df = pd.DataFrame(self.log.to_dict())
        df.to_csv(self._log_file)

        # plot metrics
        fig, ax = plt.subplots(1, len(self._metrics), figsize=(5*len(self._metrics), 5))

        # plot eval metrics
        for i, metric in enumerate(self._metrics):
            out_ax = ax[i] if len(self._metrics) > 1 else ax
            out_ax.plot(self.log[metric])
            out_ax.set_title(metric.upper())

        # finish plot
        plt.suptitle(f"Training Progress ({len(self.log[self._metrics[0]])} steps)")
        plt.tight_layout()
        plt.savefig(self._progress_file)
        plt.close()

        self.upload()
        print("Saved.")


    @torch.no_grad()
    def upload(self):
        if not constants.XLA_MAIN():
            return

        api = hf.HfApi()

        for file in [self._hyper_file, self._log_file, self._progress_file]:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=str(file).split("/")[-1],
                repo_id=self.save_repo,
                repo_type="model"
            )


    @torch.no_grad()
    def save_checkpoint(
        self,
        models
    ):
        if not constants.XLA_MAIN():
            return
        print("Saving checkpoint...")

        api = hf.HfApi()

        for name, tup in models.items():
            model, on_device = tup

            path = os.path.join(constants.LOCAL_DATA_PATH, name)

            if on_device:
                os.makedirs(path, exist_ok=True)
                model.config.save_pretrained(path, push_to_hub=False)
                xm.save(model.state_dict(), os.path.join(path, "state_dict.pt"))

            else:
                model.save_pretrained(path, push_to_hub=False)

            api.upload_folder(
                repo_id=self.save_repo,
                folder_path=path,
                path_in_repo=name,
                repo_type="model"
            )
        
        print("Saved checkpoint.")
