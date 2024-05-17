""" Models """

from models.base import BaseConfig, BaseLmModel
from models.monarc import MonArcConfig, MonArcLmModel


CONFIG_DICT = {
    "base": BaseConfig,
    "monarc": MonArcConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "monarc": MonArcLmModel,
}