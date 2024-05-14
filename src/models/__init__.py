""" Models """

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig

from models.arc import ArcLmModel

from models.base import BaseConfig, BaseLmModel


CONFIG_DICT = {
    "annelid": AnnelidConfig,
    "arc": BaseConfig,
    "base": BaseConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
    "arc": ArcLmModel,
    "base": BaseLmModel,
}