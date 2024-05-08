import torch

from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from loaders.wds_loader import get_wds_loader
from transformers import AutoTokenizer

import os


def _mp_fn(index):
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    loader = get_wds_loader("fw-4b", "train", tokenizer, 1024, True, 1)

    model = torch.nn.Embedding(100000, 2).to(xm.xla_device())
    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4)

    for x in loader:

      optimizer.zero_grad()

      x = torch.zeros(1, 2, 3, device=xm.xla_device())
      print(x.shape)

      loss = model(x).mean()

      loss.backward()
      xm.optimizer_step(optimizer)


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
