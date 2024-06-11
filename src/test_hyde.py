import torch

from transformers import AutoTokenizer

from models.hyde import HydeLmModel, HydeConfig
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


HYDE_CONFIG = "mini-hyde"
BASE_CONFIG = "mini-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    seg_ids = torch.zeros_like(x)
    seg_ids[0, 8:] = 1
    seg_ids[1, 4:6] = 1

    print("loading hyde...")
    hyde_model_config = load_model_config(HYDE_CONFIG, tokenizer)

    hyde_config = HydeConfig(**hyde_model_config)
    hyde_model = HydeLmModel(hyde_config)

    print("loading base...")
    base_model_config = load_model_config(BASE_CONFIG, tokenizer)

    base_config = BaseConfig(**base_model_config)
    base_model = BaseLmModel(base_config)

    print(f"Hyde Parameters: {sum(p.numel() for p in hyde_model.parameters())}")
    print(f"Base Parameters: {sum(p.numel() for p in base_model.parameters())}")

    hyde_out = base_model(x, segment_ids=seg_ids)

    print(f"Output mean: {hyde_out.mean().item()}")
    

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()