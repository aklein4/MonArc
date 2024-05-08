import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import argparse

from transformers import AutoTokenizer

from loaders.wds_loader import get_wds_loader
from annelid.configuration_annelid import AnnelidConfig 
from annelid.modeling_annelid import AnnelidLMModel
from training.xla_trainer import XLATrainer

import utils.constants as constants
from utils.data_utils import DotDict
from utils.config_utils import load_model_config, load_train_config


def _mp_fn(index, args):
    args = DotDict().from_dict(args)

    torch.set_default_dtype(torch.float32)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("Loading configs...")
    model_config = load_model_config(args.model_config, tokenizer)
    train_config = load_train_config(args.train_config)

    seq_length = model_config["max_position_embeddings"]

    print("Loading model...")
    annelid_config = AnnelidConfig(**model_config)
    model = AnnelidLMModel(annelid_config).to(constants.XLA_DEVICE())

    print("Loading data...")
    loader = get_wds_loader(
        args.dataset,
        "train",
        tokenizer.pad_token_id,
        seq_length,
        train_config["bs"],
        train_config["mini_bs"]
    )

    print("Train!")
    trainer = XLATrainer(
        args.save_name,
        train_config
    )
    trainer.train(
        model,
        tokenizer,
        loader
    )


if __name__ == '__main__':
    os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
    args = argparse.ArgumentParser()
    args.add_argument("--save_name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--dataset", type=str, required=True)
    args.parse_args()

    d = {}
    for k, v in vars(args).items():
        d[k] = v

    xmp.spawn(_mp_fn, args=(d),)
