import torch

from transformers import AutoTokenizer

from models.arc import ArcLmModel
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


ARC_CONFIG = "test-lm"
BASE_CONFIG = "test-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    seg_ids = torch.zeros_like(x)
    seg_ids[0, 8:] = 1
    seg_ids[1, 4:6] = 1

    print("loading arc...")
    arc_model_config = load_model_config(ARC_CONFIG, tokenizer)

    arc_config = BaseConfig(**arc_model_config)
    arc_model = ArcLmModel(arc_config)

    print("loading base...")
    base_model_config = load_model_config(BASE_CONFIG, tokenizer)

    base_config = BaseConfig(**base_model_config)
    base_model = BaseLmModel(base_config)

    print("copying weights...")
    arc_model.load_state_dict(base_model.state_dict(), strict=False)

    base_out = base_model(x, segment_ids=seg_ids)
    arc_out, debug_true, debug_false = arc_model(x, segment_ids=seg_ids, debug=True)
    arc_out, sample_true, sample_false = arc_model(x, segment_ids=seg_ids, debug=False)

    diff = torch.abs(base_out - arc_out).max()
    print(f"Arc LM vs Base LM: {diff}")

    diff = torch.abs(debug_true - debug_false).max()
    print(f"Debug true vs false: {diff}")

    diff = torch.abs(sample_true - sample_false).max()
    print(f"Sample true vs false: {diff}")
    

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()