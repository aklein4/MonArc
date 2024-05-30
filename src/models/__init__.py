""" Models """

from models.base import BaseConfig, BaseLmModel
from models.annelid import AnnelidConfig, AnnelidLmModel
from models.arc import ArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "annelid": AnnelidLmModel,
    "arc": ArcLmModel,
}
