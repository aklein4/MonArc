from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from models.base import (
    BaseConfig,
    BaseModel,
)
from models.arc import ArcTransformer
from utils.data_utils import DotDict
from utils.logging_utils import log_print


class EmbArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = ArcTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # arc modeling
        self.forward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.backward_head = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # fast sampling info
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False
    ) -> DotDict:
        """ Forward pass of the LM for training. 
         - creates negative samples
         - returns lm logits and arc predictions
         - 1 in arc predictions is fake, 0 is real, -1 is padding
         
        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length].

        Returns:
            DotDict:
                torch.Tensor: log-softmaxed logits [bs, seq_length, vocab_size]
                torch.Tensor: arc predictions [bs, seq_length-2]
                torch.Tensor: arc targets [bs, seq_length-2]
        """
        batch_size, seq_length = input_ids.shape

        # get transformer output
        true_states, memory = self.model(
            input_ids,
            segment_ids=segment_ids,
        )

        # get lm logits
        lm_logits = self.lm_head(true_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            factored_probs = torch.softmax(
                lm_logits.detach().float(), dim=-1
            ).view(-1, self.vocab_factor, self.vocab_chunk)

            outer_probs = factored_probs.sum(dim=-1)
            outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

            ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
            inner_probs = factored_probs[ar, outer_sample]
            inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

            sample = (self.vocab_chunk*outer_sample + inner_sample).view(batch_size, seq_length)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # get fake outputs
        fake_states, _ = self.model(
            fake_ids,
            segment_ids=segment_ids,
            memory=memory,
        )

        # get arc predictions
        forward_embs = self.forward_head(true_states[:, :-1])
        backward_true = self.backward_head(true_states[:, 1:])
        backward_fake = self.backward_head(fake_states[:, 1:])

        true_arc = torch.zeros(batch_size, seq_length, dtype=forward_embs.dtype, device=input_ids.device)
        fake_arc = torch.zeros(batch_size, seq_length, dtype=forward_embs.dtype, device=input_ids.device)

        true_arc[:, :-1] = (forward_embs * backward_true).sum(dim=-1)
        fake_arc[:, :-1] = (forward_embs * backward_fake).sum(dim=-1)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
