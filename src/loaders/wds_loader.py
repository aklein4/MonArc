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
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    
    def _load_data(self, data):
        stream = io.BytesIO(data)
        arr = np.lib.format.read_array(stream)
        return torch.from_numpy(arr.astype(np.int64)).long()


    def __call__(self, data):
        input_ids = [x['input_ids.npy'] for x in data]
        input_ids = [self._load_data(x) for x in input_ids]
        
        out = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        if out.shape[1] < self.max_length:
            out = F.pad(
                out,
                (0, self.max_length - out.shape[1]),
                value=self.tokenizer.pad_token_id
            )
        elif out.shape[1] > self.max_length:
            out = out[:, :self.max_length]
        
        return out
    

def get_data_files(name):
    data_files = {}
    for split in ["train", "val", "test"]:
        data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"
    return data_files


def get_wds_loader(name, split, tokenizer, max_length, parallel, bs):
    dataset = datasets.load_dataset("webdataset", data_files=get_data_files(name), split=split, streaming=True)
    collator = Collator(tokenizer, max_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        collate_fn=collator,
    )

    wrapper_type = pl.MpDeviceLoader if parallel else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE)

    return xm_loader
