import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm

import huggingface_hub as hf
from transformers import AutoTokenizer

from models.arc_old.configuration_arc import ArcConfig
from models.arc_old.modeling_arc import ArcLMModel

import utils.constants as constants


CHECKPOINT_REPO = "aklein4/Arc_mini-lm"
CHECKPOINT_SUBFOLDER = "000000010000/model"


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    config = ArcConfig.from_pretrained(CHECKPOINT_REPO, subfolder=CHECKPOINT_SUBFOLDER)
    model = ArcLMModel(config)

    local_dir = os.path.join(constants.LOCAL_DATA_PATH, CHECKPOINT_REPO.split("/")[1])
    path = hf.hf_hub_download(
        CHECKPOINT_REPO,
        subfolder=CHECKPOINT_SUBFOLDER,
        filename="state_dict.pt",
        local_dir=local_dir
    )
    model.load_state_dict(torch.load(path), strict=False)

    x = tokenizer(["""The NFL draft and a majority of free agency is in the books for 2015. That means it's time for early projections and predictions for this upcoming season. Last week, many Dolphins fans were disappointed with ESPN.com's Power Rankings. Our expect panel rated the Miami Dolphins as a middle-of-the pack team at No. 15, which is pretty much the equivalent of another .500 season. Miami made a lot of roster improvements to make a playoff push, including the $114 million signing of Pro Bowl defensive tackle Ndamukong Suh. However, another reputable source is projecting the Dolphins to do great things in 2015. According to Football Outsiders, Miami is a "hot sleeper Super Bowl contender." FBO, using its metrics and also anticipating a Tom Brady suspension, predicts the Dolphins will go 11-5 in the AFC East. Do Miami fans agree or disagree? The Dolphins have talent and filled a lot of holes this offseason. Can they post a winning season and get to the playoffs for the first time since 2008? Share your thoughts in the comment section below or send me a message via Twitter @JamesWalkerNFL. I'm curious to hear your thoughts on Miami potentially going 11-5 this upcoming season."""], return_tensors="pt", truncation=True, max_length=1024).input_ids

    lm_out = model.get_lm_logits(x, tokenizer.pad_token_id)
    true_residuals = lm_out.residuals[0]
    
    dist = torch.distributions.Categorical(logits=lm_out.lm_logits)
    samples = dist.sample((16,))[:, 0]
    residuals = model.get_arc_preds(x, samples, lm_out.kv)

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

    logp = -F.cross_entropy(
        lm_out.lm_logits[0, :-1],
        x[0, 1:],
        reduction="none"
    )
    print("Baseline PPL:", -logp.mean().item(),  torch.exp(-logp.mean()).item())

    logp -= true_residuals[1:]
    upper_logp = logp - upper[1:]
    lower_logp = logp - lower[1:]

    print("upper PPL:", -upper_logp.mean().item(), torch.exp(-upper_logp.mean()).item())
    print("lower PPL:", -lower_logp.mean().item(), torch.exp(-lower_logp.mean()).item())


if __name__ == '__main__':
    with torch.no_grad():
        main()