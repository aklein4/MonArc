
import numpy as np
import os
import webdataset as wds
from tqdm import tqdm

import huggingface_hub as hf

import utils.constants as constants


TEMP_PATH = "temp.tar.gz"

MAX_FILES_IN_SHARD = 1e12
MAX_SHARD_SIZE = 4e9

Q_SIZE = 1028*8

MIN_INTERVAL = 60


class HfShardWriter(wds.ShardWriter):
    def __init__(self, out_repo, out_path, temp_path=TEMP_PATH, *args, **kwargs):
        self.out_repo = f'{constants.HF_ID}/{out_repo}'
        self.out_path = out_path

        self.temp_path = temp_path
        self.api = hf.HfApi()
        
        hf.create_repo(
            self.out_repo,
            private=True,
            repo_type="dataset",
            exist_ok=True,
        )

        kwargs["maxcount"] = MAX_FILES_IN_SHARD
        kwargs["maxsize"] = MAX_SHARD_SIZE
        super().__init__("", *args, **kwargs)


    def next_stream(self):
        """Close the current stream and move to the next."""
        self.finish()
        if self.fname is None:
            self.fname = self.temp_path
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

    
    def finish(self):
      super().finish()
      if self.fname is not None and os.path.exists(self.fname):

        self.api.upload_file(
            repo_id=self.out_repo,
            path_or_fileobj=self.fname,
            path_in_repo=os.path.join(self.out_path, f"{self.shard:012d}.tar.gz"),
            repo_type="dataset"
        )

        os.remove(self.fname)


class TokenPackingQueue:
   
    def __init__(self, tokenizer, max_length, min_length, q_size=Q_SIZE):
        self.tokenizer = tokenizer
        self.q_size = q_size
        self.max_length = max_length
        self.min_length = min_length
    
        self.queue = [None] * q_size
        self.id_queue = [None] * q_size
        self.counts = np.array([0] * q_size)
        self.filled = np.array([False] * q_size)

        self.total_count = 0
        self.trunc_count = 0


    def _pop(self, ind):
        assert self.queue[ind] is not None
        assert self.id_queue[ind] is not None
        assert self.filled[ind]

        x = self.queue[ind]
        ids = self.id_queue[ind]

        self.queue[ind] = None
        self.id_queue[ind] = None
        self.counts[ind] = 0
        self.filled[ind] = False   

        return x, ids


    def _push(self, x, ids, ind):
        assert self.queue[ind] is None
        assert self.id_queue[ind] is None
        assert self.counts[ind] == 0
        assert not self.filled[ind]

        self.queue[ind] = x
        self.id_queue[ind] = ids
        self.counts[ind] = x.size
        self.filled[ind] = True 


    def __call__(self, x):

        # add eos token
        x = np.concatenate([x, np.array([self.tokenizer.eos_token_id], dtype=x.dtype)])
        x = x[:self.max_length]

        # ids init to zero
        ids = np.zeros_like(x)

        # only work qith small sequences
        if x.size < self.min_length:

            # try to combine from queue
            new_sizes = self.counts + x.size
            good = np.logical_and(self.filled, new_sizes <= self.max_length)
            if np.any(good):

                # get the biggest combination
                new_sizes[~good] = -1
                ind = np.argmax(new_sizes)
                
                # get new x
                x = np.concatenate(
                    [self.queue[ind], x],
                    axis=-1
                )
                ids = np.concatenate(
                    [self.id_queue[ind], ids+self.id_queue[ind][-1]+1],
                    axis=-1
                )

                # pop from queue
                self._pop(ind)

            # still not big enough, add to queue
            if x.size < self.min_length:

                # pop the largest from queue if fill
                if self.filled.all():
                    max_ind = np.argmax(self.counts)

                    y, y_ids = self._pop(max_ind)
                else:
                    y, y_ids = None, None
                
                # push to queue
                avail = np.argmin(self.filled)
                self._push(x, ids, avail)

                # restore possibly popped
                x, ids = y, y_ids

        if x is not None:
            assert x.size <= self.max_length
            assert ids.shape == x.shape

            self.total_count += self.max_length
            self.trunc_count += x.size

        return x, ids


class TokenizerMap:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __call__(self, d):
        
        # batch encode text
        input_ids = self.tokenizer(
            [t[:20*self.max_length] for t in d["text"]],
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


def create_split(
    tokenizer,
    token_iterator,
    repo,
    split,
    num_tokens,
    max_length,
    min_length,
):
    q = TokenPackingQueue(
        tokenizer=tokenizer,
        max_length=max_length,
        min_length=min_length,
    )
  
    with HfShardWriter(
        out_repo=repo,
        out_path=split,
    ) as sink:
        with tqdm(
            total=num_tokens,
            desc=split,
            mininterval=MIN_INTERVAL
        ) as pbar:

            curr_ind = 0
            while q.trunc_count < num_tokens:
                try:
                    input_ids = next(token_iterator)["input_ids"]
                except StopIteration:
                    break

                input_ids, segment_ids = q(input_ids)
                if input_ids is None:
                    continue

                sample = {
                    "__key__": f"{curr_ind:012d}",
                    "input_ids.npy": input_ids,
                    "segment_ids.npy": segment_ids
                }
                sink.write(sample)
                curr_ind += 1

                pbar.update(q.trunc_count-pbar.n)
                pbar.set_postfix(
                    ind=curr_ind,
                    perc=(q.trunc_count/q.total_count),
                    q=np.sum(q.filled),
                    q_perc=np.sum(q.filled)/q.q_size,
                    refresh=False
                )
                