import torch

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


SAMPLES = 100000


def main():
    
        lm_logits = torch.randn(15)
        vocab_factor = 5
        vocab_chunk = 3

        factored_probs = torch.softmax(
            lm_logits.detach().float(), dim=-1
        ).view(vocab_factor, vocab_chunk)

        counts = torch.zeros(15)
        for i in tqdm(range(1000)):

            outer_probs = factored_probs.sum(dim=-1)
            outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

            inner_probs = factored_probs[outer_sample]
            inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

            sample = vocab_chunk*outer_sample + inner_sample
            counts[sample] += 1

        counts /= counts.sum()

        plt.bar(range(15), counts, alpha=0.5, label='sampled')
        plt.bar(range(15), factored_probs.view(-1), alpha=0.5, label='true')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()