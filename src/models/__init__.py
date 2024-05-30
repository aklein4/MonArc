""" Models """

from models.base import BaseConfig, BaseLmModel
from models.annelid import AnnelidConfig, AnnelidLmModel
from models.arc import ArcLmModel
from models.reaper import ReaperLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
    "reaper": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "annelid": AnnelidLmModel,
    "arc": ArcLmModel,
    "reaper": ReaperLmModel,
}
