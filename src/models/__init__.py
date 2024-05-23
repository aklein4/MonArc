""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcConfig, ArcLmModel
from models.dynamarc import DynamArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": ArcConfig,
    "dynamarc": BaseConfig
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "dynamarc": DynamArcLmModel
}
