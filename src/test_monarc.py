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
    seg_ids = torch.zeros_like(x)
    seg_ids[0, 8:] = 1
    seg_ids[1, 4:6] = 1

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

    base_out = base_model(x, segment_ids=seg_ids)
    monarc_out, zero_true, zero_false = monarc_model(x, segment_ids=seg_ids, debug=False)
    
    monarc_model.control = True
    _, control_zero_true, control_zero_false = monarc_model(x, segment_ids=seg_ids, debug=False)

    monarc_model.embed_conds.weight.data.random_()
    _, control_rand_true, control_rand_false = monarc_model(x, segment_ids=seg_ids, debug=False)

    monarc_model.control = False
    _, rand_true, rand_false = monarc_model(x, segment_ids=seg_ids, debug=False)

    diff = torch.abs(base_out - monarc_out).max()
    print(f"Arc LM vs Base LM: {diff}")

    diff = torch.abs(zero_false).max()
    print(f"Zero false: {diff}")

    diff = torch.abs(zero_true - zero_false).max()
    print(f"Zero true vs false: {diff}")

    diff = max(
        torch.abs(zero_true - control_zero_true).max(),
        torch.abs(zero_false - control_zero_false).max()
    )
    print(f"Zero control vs cond: {diff}")

    diff = torch.abs(control_rand_true - control_rand_false).max()
    print(f"Control rand true vs false: {diff}")

    diff = torch.abs(control_rand_true - rand_true).max()
    print(f"Control rand true vs rand true: {diff}")

    diff = torch.abs(rand_false).max()
    print(f"Rand false: {diff}")

    diff = torch.abs(rand_true - rand_false).max()
    print(f"Rand true vs false: {diff}")
    

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()