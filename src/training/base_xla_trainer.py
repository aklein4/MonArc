import torch

import torch_xla.core.xla_model as xm

import os

import wandb
import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict
from utils.logging_utils import LogSection


class BaseXLATrainer:

    def __init__(
        self,
        project,
        name,
        config,
        debug=False
    ):
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
            base_path = os.path.join(constants.LOCAL_DATA_PATH, f"{step:012d}")

            for name, tup in models.items():
                model, on_device = tup

                path = os.path.join(base_path, name)

                if on_device:
                    os.makedirs(path, exist_ok=True)
                    xm.save(model.state_dict(), os.path.join(path, "state_dict.pt"))
                    try:
                        model.config.save_pretrained(path, push_to_hub=False)
                    except:
                        print(f"Warning: {name} config not saved")
                        pass

                else:
                    model.save_pretrained(path, push_to_hub=False)

                api.upload_folder(
                    repo_id=self.save_repo,
                    folder_path=path,
                    path_in_repo=name,
                    repo_type="model"
                )
            