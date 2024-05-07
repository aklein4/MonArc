

import os

import datasets
from transformers import GPT2TokenizerFast
import huggingface_hub as hf

import utils.constants as constants
from data_prep.token_wds import create_token_wds


TOKENIZER_URL = "openai-community/gpt2"

DATA_URL = 'HuggingFaceFW/fineweb'
DATA_SUBSET = "CC-MAIN-2024-10"

SAVE_PATH = "fw-10b" # "/home/aklein4/data/fw-10b"
SAVE_REPO = 'fw-10b'

TRAIN_SIZE = 1e10
VAL_SIZE = 1e8
TEST_SIZE = 1e8

MAX_LENGTH = 1024


def main():
    
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_URL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = datasets.load_dataset(
        DATA_URL,
        name=DATA_SUBSET,
        streaming=True,
        split="train"
    )

    check = dataset.map()

    create_token_wds(
        SAVE_PATH,
        dataset,
        tokenizer,
        TRAIN_SIZE,
        VAL_SIZE,
        TEST_SIZE,
        MAX_LENGTH
    )

    hf.create_repo(
        f"{constants.HF_ID}/{SAVE_REPO}",
        private=True,
        repo_type="dataset",
        exist_ok=True,
    )
    hf.upload_folder(
        repo_id=f"{constants.HF_ID}/{SAVE_REPO}",
        repo_type="dataset",
        folder_path=os.path.join(SAVE_PATH, "test"),
        path_in_repo="test",
    )
    hf.upload_folder(
        repo_id=f"{constants.HF_ID}/{SAVE_REPO}",
        repo_type="dataset",
        folder_path=os.path.join(SAVE_PATH, "val"),
        path_in_repo="val",
    )
    hf.upload_folder(
        repo_id=f"{constants.HF_ID}/{SAVE_REPO}",
        repo_type="dataset",
        folder_path=os.path.join(SAVE_PATH, "train"),
        path_in_repo="train",
    )


if __name__ == "__main__":
    main()