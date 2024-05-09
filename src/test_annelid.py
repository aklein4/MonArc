import torch

from transformers import AutoTokenizer

from annelid.configuration_annelid import AnnelidConfig
from annelid.modeling_annelid import AnnelidLMModel
from utils.config_utils import load_model_config, load_train_config
import utils.constants as constants


MODEL_CONFIG = "mini_quasi"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("loading model...")
    model_config = load_model_config(MODEL_CONFIG, tokenizer)

    annelid_config = AnnelidConfig(**model_config)
    model = AnnelidLMModel(annelid_config).to(constants.XLA_DEVICE())



if __name__ == '__main__':
    main()