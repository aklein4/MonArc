import torch

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

D = 10
SAMPLES = 100000


def main():
    
    logits = torch.randn(D)
    probs = torch.softmax(logits, dim=0)
    cum_probs = torch.cumsum(probs, dim=0)
    print(cum_probs)

    counts = torch.zeros(D)
    for i in tqdm(range(SAMPLES)):
        r = torch.rand(1)
        counts[(r > cum_probs).long().sum()] += 1
    counts /= SAMPLES 

    plt.bar(range(D), counts.numpy(), alpha=0.5, label="sampled")
    plt.bar(range(D), probs.numpy(), alpha=0.5, label="true")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()