
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    print("WARNING: torch_xla not found.")

import utils.constants as constants


def log_print(x):
    print(f"Device {constants.XLA_DEVICE_ID()}: {x}", flush=True)


def log_master_print(x):
    xm.master_print(f"Master: {x}", flush=True)


class LogSection:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        log_print(f"Starting {self.name}...")
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            log_print(f"Finished {self.name}.")