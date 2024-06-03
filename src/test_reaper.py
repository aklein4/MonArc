import torch

from transformers import AutoTokenizer

from models.reaper import ReaperLmModel
from models.base import BaseConfig, BaseLmModel
from utils.config_utils import load_model_config
import utils.constants as constants


ARC_CONFIG = "mini-reaper"
BASE_CONFIG = "mini-lm"


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
    arc_model = ReaperLmModel(arc_config)

    print("loading base...")
    base_model_config = load_model_config(BASE_CONFIG, tokenizer)

    base_config = BaseConfig(**base_model_config)
    base_model = BaseLmModel(base_config)

    print("copying weights...")
    arc_model.load_state_dict(base_model.state_dict(), strict=False)

    base_out = base_model(x, segment_ids=seg_ids)
    for mess in ["zeroed", "random"]:
        print(f" === {mess} === ")

        arc_out, debug_true, debug_false, debug_z = arc_model(x, segment_ids=seg_ids, debug=True)
        arc_out, sample_true, sample_false, sample_z = arc_model(x, segment_ids=seg_ids, debug=False)

        diff = torch.abs(base_out - arc_out).max()
        print(f"Arc LM vs Base LM: {diff}")

        diff = torch.abs(debug_true - debug_false).max()
        print(f"Debug true vs false: {diff}")

        diff = torch.abs(sample_true - sample_false).max()
        print(f"Sample true vs false: {diff}")

        diff = torch.abs(debug_z - sample_z).max()
        print(f"Debug z vs sample z: {diff}")

        arc_model.forward_head.weight.data.normal_()
        arc_model.l_forward_head.weight.data.normal_()
        arc_model.l_backward_head.weight.data.normal_()
        arc_model.z_head.weight.data.normal_()
    

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()