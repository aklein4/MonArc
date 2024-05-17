""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_monarc_trainer import XLAMonArcTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLAMonArcTrainer": XLAMonArcTrainer
}