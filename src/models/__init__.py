""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcConfig, ArcLmModel
from models.dynamarc import DynamArcLmModel
from models.sharc import ShArcConfig, ShArcLmModel
from models.remarc import RemArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": ArcConfig,
    "dynamarc": BaseConfig,
    "sharc": ShArcConfig,
    "remarc": ShArcConfig
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "dynamarc": DynamArcLmModel,
    "sharc": ShArcLmModel,
    "remarc": RemArcLmModel
}
