import torch

from transformers import AutoTokenizer

from models.monarc import MonArcLmModel, MonArcConfig
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
    for i in range(monarc_config.num_hidden_layers):
        monarc_model.model.layers[i].load_state_dict(base_model.model.layers[i].state_dict(), strict=True)
    for i in range(monarc_config.num_head_layers):
        monarc_model.head_model.layers[i].load_state_dict(base_model.model.layers[i+monarc_config.num_hidden_layers].state_dict(), strict=True)
    monarc_model.head_model.norm.load_state_dict(base_model.model.norm.state_dict(), strict=True)

    base_out = base_model(x, segment_ids=seg_ids)
    monarc_out, debug_true, debug_false = monarc_model(x, segment_ids=seg_ids, debug=True)
    monarc_out, sample_true, sample_false = monarc_model(x, segment_ids=seg_ids, debug=False)

    monarc_model.control = True
    monarc_out, debug_true_control, debug_false_control = monarc_model(x, segment_ids=seg_ids, debug=True)
    monarc_out, sample_true_control, sample_false_control = monarc_model(x, segment_ids=seg_ids, debug=False)

    # should be the same
    print(" ===== ")
    diff = torch.abs(base_out - monarc_out).max()
    print(f"Base diff: {diff}")

    # should both be ~zero
    print(" ===== ")
    diff = torch.abs(debug_true).max()
    print(f"Debug True diff: {diff}")
    diff = torch.abs(debug_false).max()
    print(f"Debug False diff: {diff}")

    # should both be not zero
    print(" ===== ")
    diff = torch.abs(sample_true).max()
    print(f"Sample True diff: {diff}")
    diff = torch.abs(sample_false).max()
    print(f"Sample False diff: {diff}")

    # should be the same
    print(" ===== ")
    diff = torch.abs(debug_true_control - debug_false_control).max()
    print(f"Debug Control diff: {diff}")

    # should be different
    print(" ===== ")
    diff = torch.abs(sample_true_control).max()
    print(f"Sample True Control diff: {diff}")
    diff = torch.abs(sample_false_control).max()
    print(f"Sample False Control diff: {diff}")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()