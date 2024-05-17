
import huggingface_hub as hf

import numpy as np
import pandas as pd
import os

from loaders.base_loader import BaseLoader
import utils.constants as constants


class SingleLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if f.endswith(".parquet")]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.data = None
        self.load_file(0)


    def load_file(self, file_ind):
        file = self.parquets[file_ind]

        path = os.path.join(constants.LOCAL_DATA_PATH, self.url, file)
        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path)
            
        self.data = np.array(df["text"])


    def reset(self):
        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.load_file(0)


    def __len__(self):
        return len(self.data)


    def __call__(self, batchsize):

        out = []
        while len(out) < batchsize:

            out.append(self.data[self.curr_ind])
            self.curr_ind += 1

            if self.curr_ind >= len(self.data):
                self.curr_ind = 0
                self.curr_file_ind += 1

                if self.curr_file_ind >= len(self.parquets):
                    self.curr_file_ind = 0
                    self.done = True

                self.load_file(self.curr_file_ind)

        return out
    