""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer
from trainers.xla_qlm_trainer import XLAQLMTrainer
from trainers.xla_arc_trainer import XLAArcTrainer
from trainers.xla_arc_fast_trainer import XLAArcFastTrainer
from trainers.xla_monarc_trainer import XLAMonArcTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
    "XLAQLMTrainer": XLAQLMTrainer,
    "XLAArcTrainer": XLAArcTrainer,
    "XLAArcFastTrainer": XLAArcFastTrainer,
    "XLAMonArcTrainer": XLAMonArcTrainer
}