import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm

import huggingface_hub as hf
from transformers import AutoTokenizer

from models.base import BaseConfig
from models.dynamarc import DynamArcLmModel

import utils.constants as constants
from utils.config_utils import load_model_config


CHECKPOINT_REPO = "aklein4/Arc-packed_mini-lm"
CHECKPOINT_SUBFOLDER = "000000010000/model"

CONFIG = 'mini-dynamarc'

NUM_SAMPLES = 32


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    config = BaseConfig(**load_model_config(CONFIG, tokenizer))
    model = DynamArcLmModel(config)

    local_dir = os.path.join(constants.LOCAL_DATA_PATH, CHECKPOINT_REPO.split("/")[1])
    path = hf.hf_hub_download(
        CHECKPOINT_REPO,
        subfolder=CHECKPOINT_SUBFOLDER,
        filename="state_dict.pt",
        local_dir=local_dir
    )
    model.load_state_dict(torch.load(path), strict=False)

    x = tokenizer(
        [
            """
            Decor Innovations is a leading supplier of millwork and decorative ornaments. With our extensive inventory and custom fabrication shop, we can meet your millwork needs quickly and cost effectively. Count on Decor Innovations to deliver the millwork you need, when you need it. Since ancient times, nothing has helped define spaces like the column. Its stately size and inherent strength imply a longstanding permanence to any structure - whether a single family home or a commercial building. Distinctive Rail understands the importance of selecting the right column for the right application, so whatever your project calls for, trust Distinctive Rail to have exactly what you need. Distinctive Rail is a leading provider of railing systems.
            """
        ], return_tensors="pt", truncation=True, max_length=1024
    ).input_ids
    print("NUM TOKENS:", x.shape[1])

    lm_out = model.p_lm(x)
    true_residuals = model.residuals(
        x, lm_out.true_states, memory=lm_out.memory, lm_logits=lm_out.lm_logits
    )[0]
    
    bad_samples = torch.cat(
        [model.sampler(lm_out.lm_logits) for _ in range(NUM_SAMPLES)],
        dim=0
    )
    samples = torch.zeros_like(bad_samples)
    samples[:, 1:] = bad_samples[:, :-1]

    residuals = model.residuals(
        samples,
        torch.cat([lm_out.true_states]*NUM_SAMPLES, dim=0),
        memory=torch.cat([lm_out.memory]*NUM_SAMPLES, dim=1),
        lm_logits=torch.cat([lm_out.lm_logits]*NUM_SAMPLES, dim=0)
    )

    print((true_residuals < 0).float().mean())
    print((residuals > 0).float().mean())

    lower = torch.logsumexp(-residuals - np.log(residuals.shape[0]), 0)

    exp_res = torch.exp(-residuals)
    exp_res_sum = exp_res.sum(0)
    
    uppers = []
    for i in range(samples.shape[0]):
        uppers.append(
            torch.log(
                (exp_res_sum - exp_res[i]) / (residuals.shape[0] - 1)
            )
        )
    uppers = torch.stack(uppers).mean(0)
    upper = (2*len(residuals)-1) * lower - 2*(len(residuals)-1) * uppers

    # import matplotlib.pyplot as plt
    # plt.hist(lower[:-1].numpy(), label="lower", alpha=0.5, bins=20)
    # plt.hist(upper[:-1].numpy(), label="upper", alpha=0.5, bins=20)
    print(lower[:-1].mean(), upper[:-1].mean(), -true_residuals[:-1].mean())
    # plt.legend()
    # plt.show()

    logp = -F.cross_entropy(
        lm_out.lm_logits[0, :-1],
        x[0, 1:],
        reduction="none"
    )
    print("Baseline PPL:", -logp.mean().item(),  torch.exp(-logp.mean()).item())

    true_logp = logp - true_residuals[:-1]
    upper_logp = true_logp - upper[:-1]
    lower_logp = true_logp - lower[:-1]

    print("upper PPL:", -upper_logp.mean().item(), torch.exp(-upper_logp.mean()).item())
    print("lower PPL:", -lower_logp.mean().item(), torch.exp(-lower_logp.mean()).item())


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    with torch.no_grad():
        main()