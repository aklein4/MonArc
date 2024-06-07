""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
}
