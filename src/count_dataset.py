from typing import List, Dict

import torch

import io
import numpy as np
import datasets
from tqdm import tqdm

import utils.constants as constants
    

NAME = "fw-50b"


def load_data(data):
    stream = io.BytesIO(data)
    return np.lib.format.read_array(stream)


def collate(
    data
):
    input_ids = [x['input_ids.npy'] for x in data]
    return [load_data(x).size for x in input_ids]
    

def get_data_files(
    name: str
) -> Dict[str, str]:
    data_files = {}
    for split in ["train", "val", "test"]:

        data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"
    
    return data_files


def main():
    # get streaming dataset
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=get_data_files(NAME),
        split="train", streaming=True
    )

    # wrap in loader with collator
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate,
    )

    total = 0
    pbar = tqdm(total=1)
    for x in loader:
        for c in x:
            total += c
            pbar.update(1)

    print(f"Total Tokens: {total}")


if __name__ == '__main__':
    main()