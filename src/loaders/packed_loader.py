from typing import List, Dict

import torch
import torch.nn.functional as F

import torch_xla.distributed.parallel_loader as pl

import io
import numpy as np
import datasets
import huggingface_hub as hf

import utils.constants as constants


class PackedCollator:
    def __init__(
        self,
        pad_token_id: int,
        seq_length: int
    ):
        """ A collator for WebDataset data.
         - always returns a tensor of shape (bs, seq_length)

        Args:
            pad_token_id (int): _description_
            seq_length (int): _description_
        """
        self.pad_token_id = pad_token_id
        self.seq_length = seq_length

    
    def _load_data(
        self,
        data: bytes
    ) -> torch.LongTensor:
        """ Convert the data from a byte stream to a tensor.
         - see npy_loads() in https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py

        Args:
            data (bytes): bytes representing a saved numpy array

        Returns:
            torch.LongTensor: the data as a long tensor
        """
        stream = io.BytesIO(data)
        arr = np.lib.format.read_array(stream)
        return torch.from_numpy(arr.astype(np.int64)).long()


    def __call__(
        self,
        data: List[Dict[str, bytes]]
    ) -> torch.LongTensor:
        """ Collate the data from a list of webddaataset entries
        into a single input_ids LongTensor of shape (bs, seq_length).
         - see data_prep.token_wds._extract_data() for entry format

        Args:
            data (List[Dict[str, bytes]]): list of webdataset entries

        Returns:
            torch.LongTensor: input_ids long tensor of shape (bs, seq_length)
        """

        # get list tensors
        input_ids = [x['input_ids.npy'] for x in data]
        input_ids = [self._load_data(x) for x in input_ids]
        
        seg_ids = [x['segment_ids.npy'] for x in data]
        seg_ids = [self._load_data(x) for x in seg_ids]

        # pad into single tensor
        out = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        seg_out = torch.nn.utils.rnn.pad_sequence(
            seg_ids,
            batch_first=True,
            padding_value=0
        )

        # apply seq_length constraint
        if out.shape[1] < self.seq_length:
            out = F.pad(
                out,
                (0, self.seq_length - out.shape[1]),
                value=self.pad_token_id
            )
        elif out.shape[1] > self.seq_length:
            out = out[:, :self.seq_length]

        if seg_out.shape[1] < self.seq_length:
            seg_out = F.pad(
                seg_out,
                (0, self.seq_length - seg_out.shape[1]),
                value=0
            )
        elif seg_out.shape[1] > self.seq_length:
            seg_out = seg_out[:, :self.seq_length]
        
        return out, seg_out
    

def _get_data_files(
    name: str,
    start_seq_ind: int = 0,
) -> Dict[str, str]:
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     - see data_prep.token_wds for repo layout
     
    Args:
        name (str): name of the repo to load
        start_seq_ind (int): index to start at. Default 0.

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    fs = hf.HfFileSystem()

    data_files = {}
    for split in ["train", "val", "test"]:

        if split == "train" and start_seq_ind != 0:
            avail_files = fs.ls(f"datasets/{constants.HF_ID}/{name}/{split}", detail=False)

            out = []
            for f in avail_files:

                num = int(f.split("/")[-1].split(".")[0])
                if num >= (start_seq_ind/1000000)*1.2:
                    
                    fname = f.split("/")[-1]
                    out.append(f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/{fname}")
            
            data_files[split] = out

        else:
            data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"

    return data_files


def get_packed_loader(
    name: str,
    split: str,
    pad_token_id: int,
    seq_length: int,
    bs: int,
    mini_bs: int,
    start_seq_ind: int = 0
):
    """ Get an xla token dataloader for the given wds dataset split.

    Args:
        name (str): Name of the repo to load
        split (str): split in ["train", "val", "test"]
        pad_token_id (int): pad token for collator
        max_length (int): fixed sequence length
        bs (int): batch size
        mini_bs (int): mini batch size
        start_seq_ind (int): index to start at. Default 0.
        
    Returns:
        pl.ParallelLoader: xla dataloader
    """

    # prepare batch sizes
    total_mini_bs = mini_bs * constants.NUM_XLA_DEVICES()
    if bs % total_mini_bs != 0:
        raise ValueError(f"Batch size {bs} not divisible by total mini batch size {total_mini_bs}")
    sample_size = mini_bs * (bs // total_mini_bs)

    # get streaming dataset
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=_get_data_files(name, start_seq_ind),
        split=split, streaming=True
    )

    # wrap in loader with collator
    collator = PackedCollator(pad_token_id, seq_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=collator,
        drop_last=True
    )

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader
