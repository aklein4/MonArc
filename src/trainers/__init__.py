""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_qlm_trainer import XLAQLMTrainer
from trainers.xla_arc_trainer import XLAArcTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLAQLMTrainer": XLAQLMTrainer,
    "XLAArcTrainer": XLAArcTrainer,
}