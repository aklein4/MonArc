""" Models """

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig


CONFIG_DICT = {
    "annelid": AnnelidConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
}