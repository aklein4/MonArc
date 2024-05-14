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


CHECKPOINT_REPO = f"{constants.HF_ID}/Arc_mini-lm"
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

    x = tokenizer(["""The basic goal of the effective altruism movement is to create efficient philanthropic change by backing programs and innovations that are cost-effective so that each dollar given impacts as many people as possible. The underlying tenet is that donor dollars are a limited resource, but dollars are just one of the limiting factors. There’s still another major resource that needs to be accounted for: research time. There’s a learning curve for calculation-driven cause groups (and donors) to figure out what world-plaguing problems really are the most pressing, what solutions seem the most promising or neglected, and what else might need to be done. The problem is there hasn’t been a single resource for accessing all this information in one place. To change that, Rethink Priorities, an initiative of the effective altruism awareness and engagement building nonprofit Rethink Charity, has launched Priority Wiki, a publicly editable Wikipedia-like online encyclopedia for cause prioritization wonks. It collects and categorizes vetted research around pressing charitable causes and potential interventions. “This is a big problem because thousands of hours are going into this kind of research, and you don’t want people to forget it exists, or maybe try to duplicate efforts, or just not even remember it,” says Peter Hurford, who codeveloped the wiki alongside colleague Marcus Davis. “We’re trying to capture all relevant research under a wide variety of global issues so that everyone can have a go-to spot to get up to speed.” To do that, Wiki is organized into six broad types of causes. That includes “Existential/Catastrophic Future Risks,” “Improving Research,” “Decisions and Values,” “Improving Policy,” “Developing World Health and Economic Development,” “Developed World Health and Economic Development,” and “Specific Scientific Research.” Each entry is then comprised of related topics. Under the catastrophe heading, for instance, there’s biosecurity, nuclear security, climate change, and geomagnetic storms. As the developers explain in an open letter about their efforts, the wiki is currently populated with a collection of research by effective altruism research organizations including Open Philanthropy, GiveWell, 80,000 Hours, and Animal Charity Evaluators. Many of these are formatted in what’s commonly referred to as a “shallow review,” or high-level overview of each issue, and various important statistics and findings. “That gives you a lot of opportunities to dive into the problem and make a more structured way than dumping someone a 60-item reading list,” says Hurford. Contributors are already revising the content and sharing data about things the originators hadn’t considered. Two recent additions include information about psychedelics and drug reform, and how to prevent or reduce aging-related diseases to extend our natural lifespan."""], return_tensors="pt", truncation=True, max_length=1024).input_ids

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
    print("Baseline PPL:", torch.exp(-logp.mean()).item())

    logp -= true_residuals[1:]
    upper_logp = logp - upper[1:]
    lower_logp = logp - lower[1:]

    print("upper PPL:", torch.exp(-upper_logp.mean()).item())
    print("lower PPL:", torch.exp(-lower_logp.mean()).item())


if __name__ == '__main__':
    with torch.no_grad():
        main()