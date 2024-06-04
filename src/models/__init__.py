""" Models """

from models.base import BaseConfig, BaseLmModel
from models.annelid import AnnelidConfig, AnnelidLmModel
from models.arc import ArcLmModel
from models.reaper import ReaperConfig, ReaperLmModel
from models.cortex import CortexLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
    "reaper": ReaperConfig,
    "cortex": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "annelid": AnnelidLmModel,
    "arc": ArcLmModel,
    "reaper": ReaperLmModel,
    "cortex": CortexLmModel,
}
