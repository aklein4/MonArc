import torch
import torch.nn as nn

from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from loaders.wds_loader import get_wds_loader
from transformers import AutoTokenizer
import utils.constants as constants

import os


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(60000, 128)
    
    def forward(self, x):
        return self.emb(x)


def _mp_fn(index):
    torch.set_default_dtype(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    loader = get_wds_loader("fw-4b", "train", tokenizer, 1024, True, 1)

    model = TestModel().to(xm.xla_device())

    for p in model.parameters():
      p.requires_grad = True
    model.train()

    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4)

    for x in loader:

      optimizer.zero_grad()

      loss = model(x).mean()

      loss.backward()
      xm.optimizer_step(optimizer)


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
