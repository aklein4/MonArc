import torch
import torch_xla.core.xla_model as xm

# # best device
DEVICE = "xla" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
XLA_DEVICE = None # xm.xla_device()
def _init_xla():
    global XLA_DEVICE
    XLA_DEVICE = xm.xla_device()

# local data path
LOCAL_DATA_PATH = "./local_data"

# huggingface login id
HF_ID = "aklein4"
HF_TOKEN = None
