import torch

import os

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print("Warning: torch_xla not found")

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# best device
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
XLA_DEVICE = lambda: xm.xla_device()

# id of the current device
XLA_DEVICE_ID = lambda: xm.get_ordinal()

# number of devices
NUM_XLA_DEVICES = lambda: xm.xrt_world_size()

# whether this is the main process
XLA_MAIN = lambda: xm.is_master_ordinal(local=False)

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
