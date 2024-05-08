import torch
import torch.nn as nn

from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from loaders.wds_loader import get_wds_loader
from transformers import AutoTokenizer
import utils.constants as constants

from annelid.configuration_annelid import AnnelidConfig 
from annelid.modeling_annelid import AnnelidLMModel

import os


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(60000, 128)
    
    def forward(self, x):
        return self.emb(x)


def _mp_fn(index):
    torch.set_default_dtype(torch.float32)
    constants._init_xla()
    
    MODEL_CONFIG = {
        "model_type": "annelid",
        "architectures": [
            "AnnelidLMModel"
        ],

        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "hidden_act": "silu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 128*3,
        "max_position_embeddings": 1024,
        "layer_norm_eps": 1e-05,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000,
        "tie_word_embeddings": False,

        "vocab_size": 60000, # 50258, # with padding token

        "is_prefix_lm": False,
        "is_quasi_lm": False,
        "segment_size": 32,
        "use_segment_embeds": False,

        "_attn_implementation": "sdpa",
    }

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    loader = get_wds_loader("fw-4b", "train", tokenizer, 1024, True, 1)

    config = AnnelidConfig(**MODEL_CONFIG)
    # model = AnnelidLMModel(config).to(constants.XLA_DEVICE)
    model = TestModel().to(constants.XLA_DEVICE)

    for p in model.parameters():
      p.requires_grad = True
    model.train()

    optimizer = syncfree.AdamW(model.parameters(), lr=1e-4)

    for x in loader:

      optimizer.zero_grad()

      loss = model(x, 1, 1024).mean()

      loss.backward()
      xm.optimizer_step(optimizer)
      print("Next!")


if __name__ == '__main__':
  os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
  
  xmp.spawn(_mp_fn)
