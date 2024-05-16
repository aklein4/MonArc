import torch

from transformers import AutoTokenizer

from models.monarc import MonArcConfig, MonArcLmModel
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MONARC_CONFIG = "test-monarc"
BASE_CONFIG = "test-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

    print("loading monarc...")
    monarc_model_config = load_model_config(MONARC_CONFIG, tokenizer)

    monarc_config = MonArcConfig(**monarc_model_config)
    monarc_model = MonArcLmModel(monarc_config)

    print("loading base...")
    base_model_config = load_model_config(BASE_CONFIG, tokenizer)

    base_config = BaseConfig(**base_model_config)
    base_model = BaseLmModel(base_config)

    print("copying weights...")
    monarc_model.load_state_dict(base_model.state_dict(), strict=False)
    monarc_model.norm.load_state_dict(base_model.model.norm.state_dict(), strict=True)

    base_out = base_model(x)
    monarc_out = monarc_model(x)[0]

    base_out[:, :, tokenizer.pad_token_id] = 0
    monarc_out[:, :, tokenizer.pad_token_id] = 0

    diff = torch.abs(base_out - monarc_out).max()
    print(f"Max diff: {diff}")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()