import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os


def _mp_fn(index):
    x = torch.zeros(1, 2, 3, device=xm.xla_device())
    print(x.shape)


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
