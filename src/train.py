import torch

from transformers import AutoTokenizer

from loaders.single_loader import SingleLoader

from annelid.configuration_annelid import AnnelidConfig 
from annelid.modeling_annelid import AnnelidLMModel
from training.lm_trainer import LMTrainer

import utils.constants as constants


TOKENIZER_URL = "openai-community/gpt2"
DATA_URL = 'JeanKaddour/minipile' # 'EleutherAI/the_pile_deduplicated'

NAME = "annelid-quick"

TRAIN_CONFIG = {
    "dtype": torch.bfloat16,
    "lr": 1e-3,
    "bs": 8,
    "num_steps": 1000,
    "accum_steps": 512//8,
    "warmup_steps": 1,
    "checkpoint_freq": 500,
}


MODEL_CONFIG = {
    "model_type": "annelid",
    "architectures": [
        "AnneliddLMModel"
    ],

    "bos_token_id": 50256,
    "eos_token_id": 50256,
    "hidden_act": "silu",
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 768*3,
    "max_position_embeddings": 1024,
    "layer_norm_eps": 1e-05,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_key_value_heads": 12,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000,
    "tie_word_embeddings": False,

    "vocab_size": 50258, # with padding token

    "is_prefix_lm": False,
    "is_quasi_lm": True,
    "segment_size": 16,
    "use_segment_embeds": True,

    "_attn_implementation": "eager",
}


def main():
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("Loading model...")
    config = AnnelidConfig(**MODEL_CONFIG)
    model = AnnelidLMModel(config)

    model = model.to(constants.DEVICE)
    _ = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    print("Loading data...")
    loader = SingleLoader(DATA_URL, train=True, debug=False)

    print("Train!")
    trainer = LMTrainer(
        NAME,
        **TRAIN_CONFIG
    )
    trainer.train(
        tokenizer,
        model,
        loader
    )


if __name__ == "__main__":
        main()
