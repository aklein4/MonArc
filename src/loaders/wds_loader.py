import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import datasets

import io
import numpy as np

import utils.constants as constants


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    
    def _load_data(data):
        stream = io.BytesIO(data)
        arr = np.lib.format.read_array(stream)
        return torch.from_numpy(arr.astype(np.uint64))


    def __call__(self, data):
        input_ids = [x for x in data["input_ids"]]
        input_ids = [self._load_data(x) for x in input_ids]

        return torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
    

def get_data_files(name):
    data_files = {}
    for split in ["train", "val", "test"]:
        data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"
    return data_files


def get_wds_loader(name, split, tokenizer, parallel, bs):
    dataset = datasets("webdataset", data_files=get_data_files(name), split=split, streaming=True)
    collator = Collator(tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        collate_fn=collator,
    )

    wrapper_type = pl.MpDeviceLoader if parallel else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE)

    return xm_loader
