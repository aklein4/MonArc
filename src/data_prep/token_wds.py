
import os
import numpy as np
import shutil
from tqdm import tqdm

import webdataset as wds


MAX_FILES_IN_SHARD = 1e12
MAX_SHARD_SIZE = 3e9

TOKEN_BATCH_SIZE = 10000
TOKEN_PROCESSES = 4

MIN_INTERVAL = 1



class BetterShardWriter(wds.ShardWriter):
    def __init__(self, path, *args, **kwargs):
        self.path = path
        super().__init__("", *args, **kwargs)

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = os.path.join(self.path, f"{self.shard:012d}.tar")
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1
        self.tarstream = wds.TarWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0


class TokenizerMap:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, text):
        
        # batch encode text
        input_ids = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        ).input_ids
        
        assert np.max(input_ids) < 2**16, f"Input IDs are too large for uint16: {np.max(input_ids)} > {(2**16)-1}"
        input_ids = input_ids.astype(np.uint16)
        
        # convert to list
        out = []
        for curr in input_ids:
            out.append(curr[curr != self.tokenizer.pad_token_id])

        return {"input_ids": out}


def create_token_wds(
    path,
    dataset,
    tokenizer,
    train_size,
    val_size,
    test_size,
    max_length
):
    token_dataset = dataset.map(
        TokenizerMap(tokenizer, max_length),
        input_columns="text",
        keep_in_memory=True,
        batched=True,
        batch_size=TOKEN_BATCH_SIZE,
        # num_proc=TOKEN_PROCESSES,
    )
    token_iterator = iter(token_dataset)

    _extract_data(
        os.path.join(path, "test"),
        token_iterator,
        test_size,
        desc="test"
    )

    _extract_data(
        os.path.join(path, "val"),
        token_iterator,
        val_size,
        desc="val"
    )

    _extract_data(
        os.path.join(path, "train"),
        token_iterator,
        train_size,
        desc="train"
    )


def _extract_data(path, token_iterator, target_size, desc=""):
    os.makedirs(path, exist_ok=False)

    with BetterShardWriter(path, maxsize=MAX_SHARD_SIZE, maxcount=MAX_FILES_IN_SHARD) as sink:
        with tqdm(total=target_size, desc=f"TOKENIZING {desc.upper()}", mininterval=MIN_INTERVAL) as pbar:

            curr_size = 0
            curr_ind = 0
            while curr_size < target_size:
                try:
                    input_ids = next(token_iterator)["input_ids"]
                except StopIteration:
                    break

                sample = {
                    "__key__": f"{curr_ind:012d}",
                    "input_ids.npy": input_ids,
                }
                sink.write(sample)

                curr_ind += 1
                n = input_ids.shape[0]
                curr_size += n
                pbar.update(n)
                