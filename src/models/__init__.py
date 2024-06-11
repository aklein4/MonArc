""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcLmModel
from models.hyde import HydeConfig, HydeLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": BaseConfig,
    "hyde": HydeConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "hyde": HydeLmModel
}
