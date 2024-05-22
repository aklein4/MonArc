""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_arc_trainer import XLAArcTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLAArcTrainer": XLAArcTrainer,
}