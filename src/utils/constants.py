import torch
import torch_xla.core.xla_model as xm

# best device
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
XLA_DEVICE = lambda: xm.xla_device()

# number of devices
NUM_XLA_DEVICES = lambda: xm.xrt_world_size()

# local data path
LOCAL_DATA_PATH = "./local_data"

# path to config files
CONFIG_PATH = "./configs"

# huggingface login id
HF_ID = "aklein4"
HF_TOKEN = None
