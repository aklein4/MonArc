
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass # constants handles error printing

import utils.constants as constants


def log_print(x):
    """ Print a message with the device id, and flush.

    Args:
        x: message to print
    """
    print(f"Device {constants.XLA_DEVICE_ID()}: {x}", flush=True)


def log_master_print(x):
    """ Print a message from the master device, and flush.

    Args:
        x: message to print
    """
    xm.master_print(f"Master: {x}", flush=True)


class LogSection:

    def __init__(self, name: str):
        """ A context manager for logging sections of code.
         - prints a starting message when entering
         - prints a finishing message when exiting
         - prints from all devices

        Args:
            name (str): name of the section
        """
        self.name = name
    
    
    def __enter__(self):
        log_print(f"Starting {self.name}...")
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            log_print(f"Finished {self.name}.")