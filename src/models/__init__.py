""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcConfig, ArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": ArcConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
}
