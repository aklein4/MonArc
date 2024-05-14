import torch

from transformers import AutoTokenizer

from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-lm'


def main():


    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    x = x.to(constants.XLA_DEVICE())

    print("loading config...")
    config = load_model_config(MODEL_CONFIG, tokenizer)

    print("loading eager...")
    eager_model = BaseLmModel(BaseConfig(**config)).to(constants.XLA_DEVICE())

    print("loading flash attention...")
    config["_attn_implementation"] = 'flash_attention_2_xla'
    flash_model = BaseLmModel(BaseConfig(**config)).to(constants.XLA_DEVICE())

    print("copying weights...")
    flash_model.load_state_dict(eager_model.state_dict(), strict=True)

    eager_out = eager_model(x)
    flash_out = flash_model(x)

    diff = torch.abs(eager_out.lm_logits - flash_out.lm_logits).max()
    print(f"Diff: {diff.item()}", flush=True)


if __name__ == '__main__':
    main()
