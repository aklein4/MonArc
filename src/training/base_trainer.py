import torch

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict


class BaseTrainer:

    _hyper_file = os.path.join(constants.LOCAL_DATA_PATH, "hyperparams.yml")
    _log_file = os.path.join(constants.LOCAL_DATA_PATH, "log.csv")
    _progress_file = os.path.join(constants.LOCAL_DATA_PATH, "progress.png")

    _metrics = []

    def __init__(
        self,
        save_name,
        **kwargs
    ):
        self.save_name = save_name
        self.save_repo = f"{constants.HF_ID}/{save_name}"
        hf.create_repo(
            save_name, private=True, exist_ok=True
        )
        os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

        try:
            h = self._hyperparams
        except:
            raise NotImplementedError("Please define _hyperparams in your trainer!")
        try:
                m = self._metrics
        except:
            raise NotImplementedError("Please define _metrics in your trainer!")
        
        for k in h:
            setattr(self, k, kwargs[k])

        self.log = DotDict()
        for m in self._metrics:
            self.log[m] = []


    @torch.no_grad()
    def save(self):

        # save hyperparams as csv
        with open(self._hyper_file, 'w') as outfile:
            yaml.dump(
                {k: str(getattr(self, k)) for k in self._hyperparams},
                outfile,
                default_flow_style=False
            )

        df = pd.DataFrame(self.log.to_dict())
        df.to_csv(self._log_file)

        # plot metrics
        fig, ax = plt.subplots(1, len(self._metrics), figsize=(5*len(self._metrics), 5))

        # plot eval metrics
        for i, metric in enumerate(self._metrics):
            ax[i].plot(self.log[metric])
            ax[i].set_title(metric.upper())

        # finish plot
        plt.suptitle(f"Training Progress ({len(self.log[self._metrics[0]])} steps)")
        plt.tight_layout()
        plt.savefig(self._progress_file)
        plt.close()

        self.upload()


    @torch.no_grad()
    def upload(self):
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
        api = hf.HfApi()

        for name, model in models.items():
            model.save_pretrained(
                    os.path.join(constants.LOCAL_DATA_PATH, name),
                    push_to_hub=False,
            )

            api.upload_folder(
                    repo_id=self.save_repo,
                    folder_path=os.path.join(constants.LOCAL_DATA_PATH, name),
                    path_in_repo=name,
                    repo_type="model"
            )
