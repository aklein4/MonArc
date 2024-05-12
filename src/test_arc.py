import torch

from transformers import AutoTokenizer

from models.arc.configuration_arc import ArcConfig
from models.arc.modeling_arc import ArcLMModel
from models.annelid.configuration_annelid import AnnelidConfig
from models.annelid.modeling_annelid import AnnelidLMModel
from utils.config_utils import load_model_config
import utils.constants as constants


ARC_CONFIG = "mini-arc"
ANNELID_CONFIG = "mini-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("loading arc...")
    arc_model_config = load_model_config(ARC_CONFIG, tokenizer)

    arc_config = ArcConfig(**arc_model_config)
    arc_model = ArcLMModel(arc_config)

    print("loading annelid...")
    annelid_model_config = load_model_config(ANNELID_CONFIG, tokenizer)
    annelid_model_config["segment_size"] = 4

    annelid_config = AnnelidConfig(**annelid_model_config)
    annelid_model = AnnelidLMModel(annelid_config)

    print("copying weights...")
    arc_model.load_state_dict(annelid_model.state_dict(), strict=False)

    arc_model.train()
    annelid_model.train()

    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids
    
    annelid_out = annelid_model(x)
    arc_out = arc_model.train_forward(x, tokenizer.pad_token_id)

    diff = torch.abs(annelid_out.lm_logits - arc_out.lm_logits).max()
    print(diff.item())


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()