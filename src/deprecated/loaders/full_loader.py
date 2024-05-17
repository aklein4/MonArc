
import huggingface_hub as hf

import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm

from loaders.base_loader import BaseLoader
import utils.constants as constants


class FullLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if f.endswith(".parquet")]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        data = []

        for file in tqdm(self.parquets, desc="Loading", disable=debug):

            path = os.path.join(constants.LOCAL_DATA_PATH, self.url, file)
            if os.path.exists(path):
                df = pd.read_parquet(path)
            else:
                df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_parquet(path)

            data.append(np.array(df["text"]))

            if debug:
                break

        self.data = np.concatenate(data)

        self.curr_ind = 0
        self.done = False


    def reset(self):
        self.curr_ind = 0
        self.done = False


    def __len__(self):
        return len(self.data)


    def __call__(self, batchsize):

        out = []
        while len(out) < batchsize:

            out.append(self.data[self.curr_ind])
            self.curr_ind += 1
            
            if self.curr_ind >= len(self.data):
                self.curr_ind = 0
                self.done = True

        return out
    