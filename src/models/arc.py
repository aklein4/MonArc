from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache

from models.base import BaseConfig, BaseModel, BaseTransformer

from utils.data_utils import DotDict


class ArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.transfomer = BaseTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # arc modeling
        self.arc_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    @torch.no_grad()
    def _get_arc_mask(
        self,
        input_ids: torch.LongTensor,
        cached: bool,
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        # mask with no attention
        full_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)

        nw = torch.triu(full_mask, diagonal=1) # self attending
        sw = torch.triu(full_mask, diagonal=0) # cross attending
        ne = full_mask # empty
        se = ~torch.eye(seq_length, dtype=torch.bool, device=input_ids.device) # only self

        if cached:
            return torch.cat([sw, se], dim=1)        
        return torch.cat(
            [
                torch.cat([nw, ne], dim=1),
                torch.cat([sw, se], dim=1)
            ],
            dim=0
        )
        

    @torch.no_grad()
    def _get_arc_position_ids(
        self,
        input_ids: torch.LongTensor,
        cached: bool
    ) -> torch.LongTensor:
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)

        if cached:
            return position_ids
        return torch.cat(
            [position_ids, position_ids],
            dim=0
        )


    def forward(
        self,
        input_ids: torch.LongTensor,
        pad_token_id: int,
        debug: Optional[bool] = False
    ) -> DotDict:
        """ Forward pass of the LM for training. 
         - creates negative samples
         - returns lm logits and arc predictions
         - 1 in arc predictions is fake, 0 is real, -1 is padding
         
        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length].
            pad_token_id (int): id of the pad token in the vocabulary.
            debug (Optional[bool], optional): Debug mode. Defaults to False.

        Returns:
            DotDict:
                torch.Tensor: log-softmaxed logits [bs, seq_length, vocab_size]
                torch.Tensor: arc predictions [bs, seq_length-2]
                torch.Tensor: arc targets [bs, seq_length-2]
        """
        batch_size, seq_length = input_ids.shape

        # get lm predictions
        out = self.model(input_ids, kv=DynamicCache())
        lm_logits = self.lm_head(out.hidden_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)
        
        # get negative samples
        dist = torch.distributions.Categorical(logits=lm_logits)
        neg_ids = dist.sample()
        if debug:
            neg_ids = input_ids.clone()
            neg_ids[:, :-1] = input_ids[:, 1:]

        # get arc inputs
        arc_ids = torch.cat(
            [
                input_ids[:, :1],
                neg_ids[:, :-1]
            ],
            dim=1
        )
        arc_mask = self._get_arc_mask(input_ids)

        # get arc outputs
        arc_out = self.model(
            input_ids=arc_ids,
            attention_mask=arc_mask,
            kv=out.kv
        )

        arc_states = torch.cat(
            [
                out.hidden_states,
                arc_out.hidden_states
            ],
            dim=1
        )

        # get arc predictions
        # formated to use cross entropy loss
        arc_preds = self.arc_head(arc_states)[:, :, 0]
        arc_preds = torch.stack(
            [-arc_preds/2, arc_preds/2],
            dim=-1
        )

        # get arc targets
        arc_targets = torch.zeros(batch_size, 2*seq_length, dtype=input_ids.dtype, device=input_ids.device)
        arc_targets[:, :seq_length] = 1
        
        # target padding
        arc_targets[:, 0] = -1
        arc_targets[:, seq_length] = -1
        arc_targets = torch.masked_fill(
            arc_targets, 
            torch.cat([input_ids, input_ids], dim=1) == pad_token_id,
            -1
        )

        return DotDict(
            lm_logits=lm_logits,
            arc_preds=arc_preds,
            arc_targets=arc_targets
        )
