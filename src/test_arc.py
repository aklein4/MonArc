import torch

from transformers import AutoTokenizer

from models.arc.configuration_arc import ArcConfig
from models.arc.modeling_arc import ArcLMModel
from utils.config_utils import load_model_config
import utils.constants as constants


MODEL_CONFIG = "test-arc"


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("loading model...")
    model_config = load_model_config(MODEL_CONFIG, tokenizer)
    model_config["_gradient_checkpointing"] = False

    arc_config = ArcConfig(**model_config)
    model = ArcLMModel(arc_config)

    x = tokenizer("Hello, my dog is cute", return_tensors="pt", padding="max_length", max_length=16).input_ids
    
    out = model.train_forward(x, tokenizer.pad_token_id, debug=True)

    print("x:", x)
    print("arc targets:", out.arc_targets)

    pos, neg = torch.split(out.arc_preds, x.shape[-1], dim=1)
    diff = torch.max(torch.abs(pos - neg))
    print("diff:", diff)

if __name__ == '__main__':
    torch.set_printoptions(threshold=10_000)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()