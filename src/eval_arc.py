import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm

import huggingface_hub as hf
from transformers import AutoTokenizer

from models.arc.configuration_arc import ArcConfig
from models.arc.modeling_arc import ArcLMModel

import utils.constants as constants


CHECKPOINT_REPO = f"{constants.HF_ID}/Arc_mini-arc-w10"
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

    x = tokenizer(["""Dungeons & Dragons (commonly abbreviated as D&D or DnD)[2] is a fantasy tabletop role-playing game (RPG) originally created and designed by Gary Gygax and Dave Arneson.[3][4][5] The game was first published in 1974 by Tactical Studies Rules, Inc. (TSR).[5] It has been published by Wizards of the Coast, later a subsidiary of Hasbro, since 1997. The game was derived from miniature wargames, with a variation of the 1971 game Chainmail serving as the initial rule system.[4][6] D&D's publication is commonly recognized as the beginning of modern role-playing games and the role-playing game industry,[5][7] and also deeply influenced video games, especially the role-playing video game genre.[8][9][10] D&D departs from traditional wargaming by allowing each player to create their own character to play instead of a military formation. These characters embark upon adventures within a fantasy setting. A Dungeon Master (DM) serves as referee and storyteller for the game, while maintaining the setting in which the adventures occur, and playing the role of the inhabitants of the game world, known as non-player characters (NPCs). The characters form a party and they interact with the setting's inhabitants and each other. Together they solve problems, engage in battles, explore, and gather treasure and knowledge. In the process, player characters earn experience points (XP) to level up, and become increasingly powerful over a series of separate gaming sessions.[3][7][11] Players choose a class when they create their character, which gives them special perks and abilities every few levels."""], return_tensors="pt", truncation=True, max_length=1024).input_ids

    lm_out = model.get_lm_logits(x, tokenizer.pad_token_id)
    true_residuals = lm_out.residuals[0]
    
    dist = torch.distributions.Categorical(logits=lm_out.lm_logits)
    samples = dist.sample((32,))[:, 0]
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
    print("Baseline Loss:", logp.mean().item()) # torch.exp(-logp.mean()).item())

    logp -= true_residuals[1:]
    upper_logp = logp - upper[1:]
    lower_logp = logp - lower[1:]

    print("upper Loss:", upper_logp.mean().item()) # torch.exp(-upper_logp.mean()).item())
    print("lower Loss:", lower_logp.mean().item()) # torch.exp(-lower_logp.mean()).item())


if __name__ == '__main__':
    with torch.no_grad():
        main()