""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcConfig, ArcLmModel
from models.monarc import MonArcConfig, MonArcLmModel
from models.embarc import EmbArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": ArcConfig,
    "monarc": MonArcConfig,
    "embarc": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "monarc": MonArcLmModel,
    "embarc": EmbArcLmModel,
}
