""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_arc_trainer import XLAArcTrainer
from trainers.xla_monarc_trainer import XLAMonArcTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLAArcTrainer": XLAArcTrainer,
    "XLAMonArcTrainer": XLAMonArcTrainer
}