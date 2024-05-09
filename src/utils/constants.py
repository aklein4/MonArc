import torch

import os

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print("Warning: torch_xla not found")

# get the base path, first parent is utils next is src
from pathlib import Path
BASE_PATH = Path(__file__).absolute().parent().parent()

# best device
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
XLA_DEVICE = lambda: xm.xla_device()

# number of devices
NUM_XLA_DEVICES = lambda: xm.xrt_world_size()

# whether this is the main process
XLA_MAIN = lambda: xm.is_master_ordinal()

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# paths to config files
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, "model_configs")
TRAIN_CONFIG_PATH = os.path.join(BASE_PATH, "train_configs")

# gpt2 tokenizer
GPT2_TOKENIZER = 'openai-community/gpt2'

# huggingface login id
HF_ID = "aklein4"
HF_TOKEN = None
