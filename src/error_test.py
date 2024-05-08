import torch

from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from loaders.wds_loader import get_wds_loader
from transformers import AutoTokenizer
import utils.constants as constants

import os


def _mp_fn(index):
    constants._init_xla()
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    loader = get_wds_loader("fw-4b", "train", tokenizer, 1024, True, 1)

    model = torch.nn.Embedding(100000, 2).to(xm.xla_device())
    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4)

    for x in loader:

      optimizer.zero_grad()

      loss = model(x).mean()

      loss.backward()
      xm.optimizer_step(optimizer)
      print("Next!")


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
