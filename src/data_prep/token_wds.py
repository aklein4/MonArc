
import os
import numpy as np
import shutil
from tqdm import tqdm

import webdataset as wds


TMP_DIR = "__wds_tmp__"

MAX_SHARD_COUNT = 1e18
MAX_SHARD_SIZE = 1e9


class TMPManager:
    def __init__(self, path):
        self.path = path
    
    def __enter__(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.path)


class BetterShardWriter(wds.ShardWriter):
    def __init__(self, path, *args, **kwargs):
        self.path = path
        super().__init__("", *args, **kwargs)

    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        self.fname = os.path.join(self.path, f"{self.shard:09d}.tar")
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
    
    def __call__(self, d):

        input_ids = self.tokenizer(
            d["text"],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        ).input_ids[0]
        
        assert np.max(input_ids) < 2**16, f"Input IDs are too large for uint16: {np.max(input_ids)} > {(2**16)-1}"
        input_ids = input_ids.astype(np.uint16)
        
        return {"input_ids": input_ids}


def create_token_wds(
    path,
    dataset,
    tokenizer,
    val_size,
    train_size,
    max_length
):
    token_dataset = dataset.map(TokenizerMap(tokenizer, max_length))
    token_iterator = iter(token_dataset)

    with TMPManager(TMP_DIR):
        _extract_data(
            os.path.join(path, "val"),
            token_iterator,
            val_size,
            desc="Val Split"
        )

    with TMPManager(TMP_DIR):
        _extract_data(
            os.path.join(path, "train"),
            token_iterator,
            train_size,
            desc="Train Split"
        )


def _extract_data(path, token_iterator, target_size, desc=None):

    with tqdm(total=target_size, desc=f"{desc} Tokens") as pbar:

        curr_size = 0
        curr_ind = 0
        while curr_size < target_size:
            try:
                input_ids = next(token_iterator)["input_ids"]
            except StopIteration:
                break

            np.save(os.path.join(TMP_DIR, f"{desc}_{curr_ind:012d}.pt"), input_ids)

            n = input_ids.shape[0]
            curr_size += n
            pbar.update(n)
            curr_ind += 1
    
    os.makedirs(path, exist_ok=True)
    with BetterShardWriter(path, maxcount=MAX_SHARD_COUNT, maxsize=MAX_SHARD_SIZE) as sink:
        
        for f in tqdm(os.listdir(TMP_DIR), desc=f"{desc} Shards"):
            with open(os.path.join(TMP_DIR, f), "rb") as stream:

                input_ids = np.frombuffer(stream.read(), dtype=np.uint16)

                sample = {
                    "__key__": f.split(".")[0],
                    "input_ids.npy": input_ids
                }
                sink.write(sample)
