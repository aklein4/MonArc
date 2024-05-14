""" Models """

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig

from models.arc.modeling_arc import ArcLMModel
from models.arc.configuration_arc import ArcConfig

from models.base import BaseConfig, BaseLmModel


CONFIG_DICT = {
    "annelid": AnnelidConfig,
    "arc": ArcConfig,
    "base": BaseConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
    "arc": ArcLMModel,
    "base": BaseLmModel,
}