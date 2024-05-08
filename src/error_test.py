import torch

from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os


def _mp_fn(index):
    
    model = torch.nn.Linear(3, 2).to(xm.xla_device())
    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4)

    for _ in range(1000):    

      optimizer.zero_grad()

      x = torch.zeros(1, 2, 3, device=xm.xla_device())
      print(x.shape)

      loss = model(x).sum()

      loss.backward()
      xm.optimizer_step(optimizer, barrier=True)


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
