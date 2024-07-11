""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcLmModel
from models.hyde import HydeConfig, HydeLmModel
from models.forg import ForgLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": BaseConfig,
    "hyde": HydeConfig,
    "forg": BaseConfig
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "hyde": HydeLmModel,
    "forg": ForgLmModel
}
