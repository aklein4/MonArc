""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcConfig, ArcLmModel
from models.dynamarc import DynamArcLmModel
from models.sharc import ShArcConfig, ShArcLmModel
from models.remarc import RemArcLmModel
from models.annelid import AnnelidConfig, AnnelidLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": ArcConfig,
    "dynamarc": BaseConfig,
    "sharc": ShArcConfig,
    "remarc": ShArcConfig,
    "annelid": AnnelidConfig
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "dynamarc": DynamArcLmModel,
    "sharc": ShArcLmModel,
    "remarc": RemArcLmModel,
    "annelid": AnnelidLmModel
}
