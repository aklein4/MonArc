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
    x = torch.zeros(1, 2, 3, device=xm.xla_device())
    print(x.shape)


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
