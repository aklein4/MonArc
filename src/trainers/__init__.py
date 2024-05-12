""" Training package """

from .xla_lm_trainer import XLALMTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
}