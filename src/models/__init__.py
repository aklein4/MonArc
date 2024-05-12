""" Models """

from .annelid.modeling_annelid import AnnelidLMModel
from .annelid.configuration_annelid import AnnelidConfig


CONFIG_DICT = {
    "annelid": AnnelidConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
}