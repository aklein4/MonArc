from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import (
    BaseConfig,
    BaseTransformer,
    BaseModel
)

from utils.data_utils import DotDict
from utils.logging_utils import log_print
import utils.constants as constants


class AnnelidConfig(BaseConfig):

    model_type = 'annelid'

    def __init__(
        self,
        *args,
        segment_size: int = 32,
        **kwargs,
    ):
        
        self.segment_size = segment_size

        super().__init__(*args, **kwargs)


class AnnelidLmModel(BaseModel):

    def __init__(self, config: AnnelidConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # annelid info
        self.segment_size = config.segment_size

        # extra embeddings
        self.embed_types = nn.Embedding(2, config.hidden_size)
        self.embed_segments = nn.Embedding(self.segment_size, config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()


    def _annelid_mask(
        self,
        input_ids: torch.LongTensor,
    ):
        bs, seq_length = input_ids.shape
        assert seq_length % self.segment_size == 0, f"Sequence length {seq_length} must be divisible by segment size {self.segment_size}"
        n_segments = seq_length // self.segment_size

        # self attending segments
        nw = torch.ones(1, n_segments, n_segments, dtype=torch.bool, device=input_ids.device)
        nw = torch.triu(nw, diagonal=1)
        nw = torch.repeat_interleave(nw, self.segment_size, dim=1)
        nw = torch.repeat_interleave(nw, self.segment_size, dim=2)

        # segments for cross attention
        sw = torch.ones(1, n_segments, n_segments, dtype=torch.bool, device=input_ids.device)
        sw = torch.triu(sw, diagonal=0)
        sw = torch.repeat_interleave(sw, self.segment_size, dim=1)
        sw = torch.repeat_interleave(sw, self.segment_size, dim=2)

        # empty
        ne = torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)

        # auto regressive within segments
        se = torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
        se = torch.triu(se, diagonal=1)
        se = torch.logical_xor(se, ~sw)

        # combine
        return torch.cat(
            [
                torch.cat([nw, ne], dim=2),
                torch.cat([sw, se], dim=2)
            ],
            dim=1
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
    ) -> DotDict:

        # prepare inputs
        annelid_ids = torch.cat([input_ids, input_ids], dim=1)
        pos_ids = torch.cat([self.model._get_position_ids(input_ids)]*2, dim=1)
        annelid_mask = self._annelid_mask(input_ids)

        # get extras
        types = torch.zeros_like(annelid_ids)
        types[:, :input_ids.shape[1]] = 1
        extra_states = self.embed_types(types)

        seg_ids = torch.arange(annelid_ids.shape[1], device=annelid_ids.device, dtype=annelid_ids.dtype)
        seg_ids = seg_ids % self.segment_size
        extra_states = extra_states + self.embed_segments(seg_ids)[None]

        # get lm predictions
        out = self.model(
            input_ids=annelid_ids,
            position_ids=annelid_ids,
            attention_mask=annelid_mask,
            position_ids=pos_ids,
            extra_states=extra_states,
        )[:, input_ids.shape[1]:]

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
