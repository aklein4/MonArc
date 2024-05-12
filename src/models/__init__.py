""" Models """

from models.annelid.modeling_annelid import AnnelidLMModel
from models.annelid.configuration_annelid import AnnelidConfig

from models.arc.modeling_arc import ArcLMModel
from models.arc.configuration_arc import ArcConfig


CONFIG_DICT = {
    "annelid": AnnelidConfig,
    "arc": ArcConfig,
}

MODEL_DICT = {
    "annelid": AnnelidLMModel,
    "arc": ArcLMModel,
}