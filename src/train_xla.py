import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os

from transformers import AutoTokenizer

from loaders.wds_loader import get_wds_loader
from annelid.configuration_annelid import AnnelidConfig 
from annelid.modeling_annelid import AnnelidLMModel
from training.xla_trainer import XLATrainer

import utils.constants as constants


def _mp_fn(index):
    constants._init_xla()

    TOKENIZER_URL = "openai-community/gpt2"

    DATA_NAME = 'fw-4b'

    LR = 1e-3
    BS = 1

    MODEL_CONFIG = {
        "model_type": "annelid",
        "architectures": [
            "AnnelidLMModel"
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
        "is_quasi_lm": False,
        "segment_size": 32,
        "use_segment_embeds": True,

        "_attn_implementation": "sdpa",
    }
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("Loading model...")
    config = AnnelidConfig(**MODEL_CONFIG)
    model = AnnelidLMModel(config).to(constants.XLA_DEVICE)

    print("Loading data...")
    loader = get_wds_loader(DATA_NAME, "train", tokenizer, MODEL_CONFIG["max_position_embeddings"], parallel=True, bs=BS)

    print("Train!")
    trainer = XLATrainer(
        model,
        tokenizer,
        loader,
        lr=LR,
        bs=BS,
        num_steps=1000
    )
    trainer.train()


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"]="localservice;0;localhost:51011"
  xmp.spawn(_mp_fn)
