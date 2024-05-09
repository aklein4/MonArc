import torch
import torch.distributed as dist

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
from utils.logging_utils import log_print


def _mp_fn(index, args):
    args = DotDict().from_dict(args)

    # setup
    torch.set_default_dtype(torch.float32)
    dist.init_process_group('xla', init_method='xla://') # needed?
    print(
        f"XLA Master: {xm.is_master_ordinal(local=False)}, XLA Ordinal: {xm.get_ordinal()}, XLA Local Master: {xm.is_master_ordinal(local=True)}, XLA Local Ordinal: {xm.get_local_ordinal()}, XLA World Size: {xm.xrt_world_size()}, Dist Rank: {dist.get_rank()}, Dist World Size: {dist.get_world_size()}",
        flush=True
    )

    log_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    log_print("Loading configs...")
    model_config = load_model_config(args.model_config, tokenizer)
    train_config = load_train_config(args.train_config)

    seq_length = model_config["max_position_embeddings"]

    log_print("Loading model...")
    annelid_config = AnnelidConfig(**model_config)
    model = AnnelidLMModel(annelid_config).to(constants.XLA_DEVICE())
    xm.broadcast_master_param(model)

    log_print("Loading data...")
    loader = get_wds_loader(
        args.dataset,
        "train",
        tokenizer.pad_token_id,
        seq_length,
        train_config["bs"],
        train_config["mini_bs"]
    )

    log_print("Train!")
    trainer = XLATrainer(
        args.project,
        args.name,
        train_config
    )
    trainer.train(
        model,
        tokenizer,
        loader
    )


if __name__ == '__main__':
  
    # setup PJRT runtime
    os.environ['PJRT_DEVICE'] = 'TPU'

    # handle arguments
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--dataset", type=str, required=True)
    args = args.parse_args()

    # arguments must be picklable
    d = {}
    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool)):
            d[k] = v

    xmp.spawn(_mp_fn, args=(d,))
