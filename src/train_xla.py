import torch

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import argparse
import huggingface_hub as hf

from transformers import AutoTokenizer

from loaders.packed_loader import get_packed_loader
from models import CONFIG_DICT, MODEL_DICT
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_model_config, load_train_config
from utils.logging_utils import log_print


def _mp_fn(index, args):

    # setup
    torch.set_default_dtype(torch.float32)

    # debug infp
    log_print(
        f"Local Ordinal: {xm.get_local_ordinal()}, Local Master: {xm.is_master_ordinal(local=True)}, Master: {xm.is_master_ordinal(local=False)}, World Size: {xm.xrt_world_size()}"
    )

    log_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    log_print("Loading configs...")
    model_config = load_model_config(args.model_config, tokenizer)
    train_config = load_train_config(args.train_config)

    seq_length = model_config["max_position_embeddings"]
    start_seq_ind = train_config.get("start_seq_ind", 0)
    checkpoint = train_config.get("checkpoint", None)

    log_print("Loading model...")
    model_type = model_config.pop("model_type")
    model_type_config = CONFIG_DICT[model_type](**model_config)
    model = MODEL_DICT[model_type](model_type_config)
    
    if checkpoint is not None:
        log_print("Loading checkpoint...")

        repo, name = tuple(checkpoint.split("/"))
        checkpoint_local_dir = os.path.join(constants.LOCAL_DATA_PATH, "checkpoint")
        checkpoint_path = hf.hf_hub_download(
            f"{constants.HF_ID}/{repo}",
            subfolder=f"{name}/model",
            filename="state_dict.pt",
            local_dir=checkpoint_local_dir
        )

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=True)
        del checkpoint

        model = model.to(constants.XLA_DEVICE())

    elif not args.debug:
        # broadcast with bfloat16 for speed
        log_print("Syncing model...")

        model = model.to(constants.XLA_DEVICE())

        model = model.to(torch.bfloat16)
        xm.broadcast_master_param(model)
        model = model.to(torch.float32)
    
    # log_print("Compiling model...")
    # model.model = torch.compile(model.model, backend='openxla')

    log_print("Loading data...")
    loader = get_packed_loader(
        args.dataset,
        "train",
        tokenizer.pad_token_id,
        seq_length,
        train_config["bs"],
        train_config["mini_bs"],
        start_seq_ind=start_seq_ind
    )

    log_print("Train!")
    trainer_type = train_config.pop("trainer_type")
    trainer = TRAINER_DICT[trainer_type](
        args.project,
        args.name,
        train_config,
        debug=args.debug
    )
    trainer.train(
        model,
        tokenizer,
        loader
    )


if __name__ == '__main__':
  
    # setup PJRT runtime
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['XLA_NO_SPECIAL_SCALARS'] = '1'

    # handle arguments
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    xmp.spawn(_mp_fn, args=(args,))
