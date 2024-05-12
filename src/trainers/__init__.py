""" Training package """

from trainers.xla_lm_trainer import XLALMTrainer


TRAINER_DICT = {
    "XLALMTrainer": XLALMTrainer,
}