import torch

from transformers import AutoTokenizer

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = 'test-lm'


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

    print("loading base...")
    base_config = BaseConfig(**load_model_config(MODEL_CONFIG, tokenizer))
    base_model = BaseLmModel(base_config)

    print("loading annelid...")
    annelid_model_config = load_model_config(MODEL_CONFIG, tokenizer)
    annelid_model_config["segment_size"] = 4

    annelid_config = AnnelidConfig(**annelid_model_config)
    print(annelid_config.use_parallel_residual)
    annelid_model = AnnelidLMModel(annelid_config)

    print("copying weights...")
    base_model.load_state_dict(annelid_model.state_dict(), strict=True)
    
    base_model.eval()
    annelid_model.eval()

    annelid_out = annelid_model(x)
    base_out = base_model(x)

    diff = torch.abs(annelid_out.lm_logits - base_out.lm_logits).max()
    print(f"Diff: {diff}")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()