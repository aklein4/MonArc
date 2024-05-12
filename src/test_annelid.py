import torch

from transformers import AutoTokenizer

from models.annelid.configuration_annelid import AnnelidConfig
from models.annelid.modeling_annelid import AnnelidLMModel
from utils.config_utils import load_model_config, load_train_config
import utils.constants as constants


MODEL_CONFIG = "test-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("loading model...")
    model_config = load_model_config(MODEL_CONFIG, tokenizer)
    model_config["segment_size"] = 5

    annelid_config = AnnelidConfig(**model_config)
    model = AnnelidLMModel(annelid_config)

    x = torch.randint(0, 1000, (3, 20)).long()
    model(x)


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()