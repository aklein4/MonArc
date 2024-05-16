""" Models """

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig

from models.arc_old.modeling_arc import ArcLMModel
from models.arc_old.configuration_arc import ArcConfig

from models.base import BaseConfig, BaseLmModel
from models.arc import ArcLmModel
from models.monarc import MonArcConfig, MonArcLmModel

CONFIG_DICT = {
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
    "base": BaseConfig,
    "arc_old": ArcConfig,
    "monarc": MonArcConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
    "arc": ArcLmModel,
    "base": BaseLmModel,
    "arc_old": ArcLMModel,
    "monarc": MonArcLmModel,
}