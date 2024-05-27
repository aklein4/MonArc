import torch

import numpy as np
import os

from transformers import AutoTokenizer

from loaders.packed_loader import get_packed_loader


DATASET_NAME = 'fineweb-2024-packed'


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print("Loading data...")
    loader = get_packed_loader(
        DATASET_NAME,
        "train",
        tokenizer.pad_token_id,
        1024,
        2, 2, 51200000
    )


    count = 0
    with open('data_2.txt', 'w') as f:
        for x, k in loader:

            f.write(f"\nBatch {count}\n")
            for x_batch, k_batch in zip(x, k):

                for i in range(torch.max(k_batch)+1):
                    f.write(f"\nSequence {i}\n")
                    f.write(tokenizer.decode(x_batch[k_batch == i].numpy()))
                    f.write("\n")

            count += 1
            if count == 100:
                break


if __name__ == '__main__':
    main()