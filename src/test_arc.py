import torch

from transformers import AutoTokenizer

from models.arc.configuration_arc import ArcConfig
from models.arc.modeling_arc import ArcLMModel
from models.annelid.configuration_annelid import AnnelidConfig
from models.annelid.modeling_annelid import AnnelidLMModel
from utils.config_utils import load_model_config
import utils.constants as constants


ARC_CONFIG = "test-arc"
ANNELID_CONFIG = "test-lm"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    x = tokenizer(["Hello, my dog is cute", "His dog is cute too", "All dogs are cute"], return_tensors="pt", padding="max_length", max_length=16).input_ids

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
    
    annelid_out = annelid_model(x)
    negatives = arc_model.sample_negatives(x, tokenizer.pad_token_id)
    arc_out = arc_model.forward_from_sample(x, negatives, tokenizer.pad_token_id, debug=True)

    lm_diff = torch.abs(annelid_out.lm_logits - arc_out.lm_logits).max()
    print(f"LM logits diff: {lm_diff}")

    pos, neg = torch.split(arc_out.arc_preds, x.shape[1], dim=1)
    arc_diff = torch.abs(pos - neg).max()
    print(f"Arc logits diff: {arc_diff}")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()