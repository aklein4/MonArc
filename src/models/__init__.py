""" Models """

from models.base import BaseConfig, BaseLmModel
from models.annelid import AnnelidConfig, AnnelidLmModel
from models.arc import ArcLmModel
from models.reaper import ReaperConfig, ReaperLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
    "reaper": ReaperConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "annelid": AnnelidLmModel,
    "arc": ArcLmModel,
    "reaper": ReaperLmModel,
}
