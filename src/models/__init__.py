""" Models """

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcLmModel
from models.monarc import MonArcConfig, MonArcLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "arc": BaseConfig,
    "monarc": MonArcConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "arc": ArcLmModel,
    "monarc": MonArcLmModel
}
