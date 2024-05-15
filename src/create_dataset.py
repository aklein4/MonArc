
import datasets
from transformers import GPT2TokenizerFast

from data_prep.packed_data import create_split, TokenizerMap


TOKENIZER_URL = "openai-community/gpt2"

DATA_URL = 'HuggingFaceFW/fineweb'
DATA_SUBSET = "CC-MAIN-2024-10"

SAVE_REPO = 'fineweb-2024-packed'

TRAIN_SIZE = None
VAL_SIZE = 1e8
TEST_SIZE = 1e8

BATCH_SIZE = 1024*8

MIN_LENGTH = 1024-64
MAX_LENGTH = 1024


def main():
    
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_URL, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = datasets.load_dataset(
        DATA_URL,
        name=DATA_SUBSET,
        streaming=True,
        split="train"
    )
    dataset = dataset.map(
        TokenizerMap(tokenizer, MAX_LENGTH),
        batched=True,
        batch_size=BATCH_SIZE
    )
    data_iterator = iter(dataset)

    create_split(
        tokenizer,
        data_iterator,
        SAVE_REPO,
        "val",
        VAL_SIZE,
        MAX_LENGTH,
        0
    )

    create_split(
        tokenizer,
        data_iterator,
        SAVE_REPO,
        "test",
        TEST_SIZE,
        MAX_LENGTH,
        0
    )

    create_split(
        tokenizer,
        data_iterator,
        SAVE_REPO,
        "train",
        TRAIN_SIZE,
        MAX_LENGTH,
        MIN_LENGTH
    )


if __name__ == "__main__":
    main()