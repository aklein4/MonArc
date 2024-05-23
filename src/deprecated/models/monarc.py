from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.base import (
    BaseConfig,
    BaseTransformer,
    BaseModel
)
from models.arc import ArcDecoderLayer
from utils.data_utils import DotDict
from utils.model_utils import EfficientSampler
from utils.logging_utils import log_print


class MonArcConfig(BaseConfig):

    model_type = "monarc"

    def __init__(
        self,
        *args,
        num_head_layers=4,
        control=False,
        **kwargs
    ):
        
        self.num_head_layers = num_head_layers
        self.control = control

        super().__init__(*args, **kwargs)


class MonArcTransformer(BaseTransformer):

    def __init__(self, config: MonArcConfig):
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArcDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # indices for the head
        assert config.num_head_layers <= config.num_hidden_layers
        self.num_hidden_layers = config.num_hidden_layers
        self.num_head_layers = config.num_head_layers
        self.hidden_layer_ids = list(range(config.num_hidden_layers-config.num_head_layers))
        self.head_layer_ids = list(range(config.num_hidden_layers-config.num_head_layers, config.num_hidden_layers))

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        hidden_states: Optional[torch.Tensor]=None,
        input_ids: Optional[torch.LongTensor]=None,
        memory: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv=None,
    ) -> DotDict:
        assert hidden_states is not None or input_ids is not None
        assert not (hidden_states is not None and input_ids is not None)
        
        head = hidden_states is not None
        if head:
            input_ids = torch.zeros(hidden_states.shape[:-1], dtype=torch.long, device=hidden_states.device)
        else:
            hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        mem_out = []
        layer_list = self.head_layer_ids if head else self.hidden_layer_ids
        raw_idx = -1
        for layer_idx in layer_list:
            decoder_layer = self.layers[layer_idx]
            raw_idx += 1

            mem_out.append(hidden_states)
            if memory is None:
                mem_in = None
            else:
                mem_in = memory[raw_idx]

            if self.gradient_checkpointing and self.training and self.head_gradient_checkpointing:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")
                log_print("Head grad check!")

                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    mem_in,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    memory=mem_in,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=kv,
                    output_attentions=False,
                    use_cache=(kv is not None),
                )[0]

        if head:
            hidden_states = self.norm(hidden_states)

        if len(mem_out) > 0:
            mem_out = torch.stack(mem_out, dim=0)
        else:
            mem_out = None

        return hidden_states, mem_out


class MonArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = MonArcTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # arc modeling
        self.embed_conds = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_token_id)
        self.sampler = EfficientSampler(config.vocab_size)

        # extras
        self.control = config.control

        # Initialize weights and apply final processing
        self.post_init()

        # init conds to zero
        self.embed_conds.weight.data.zero_()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        debug=False
    ) -> DotDict:
        batch_size, seq_length = input_ids.shape

        # reuse the attention mask
        attention_mask = self.model._get_mask(input_ids, None, segment_ids)

        # get transformer output
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cached_mask=True
        )[0]

        # get the lm logits
        lm_states, memory = self.model(
            hidden_states=hidden_states,
            memory=None,
            attention_mask=attention_mask,
            cached_mask=True
        )
        lm_logits = self.lm_head(lm_states)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        # get the fake ids
        if debug:
            fake_ids = input_ids.clone()
        else:
            sample = self.sampler(lm_logits)
            fake_ids = input_ids.clone()
            fake_ids[:, 1:] = sample[:, :-1]

        # get the input tokens for the head
        true_tokens = torch.zeros_like(input_ids)
        fake_tokens = torch.zeros_like(fake_ids)
        if not self.control:
            true_tokens[:, :-1] = input_ids[:, 1:]
            fake_tokens[:, :-1] = fake_ids[:, 1:]

        true_states = hidden_states + self.embed_conds(true_tokens)
        fake_states = hidden_states + self.embed_conds(fake_tokens)

        # get the true fake head outputs
        true_states, fake_states = self.model(
            hidden_states=torch.cat([true_states, fake_states], dim=0),
            memory=torch.cat([memory, memory], dim=1),
            attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
            cached_mask=True
        )[0].chunk(2, dim=0)

        # get the true and fake logits
        true_logits = self.lm_head(true_states)
        true_logits = F.log_softmax(true_logits, dim=-1)

        fake_logits = self.lm_head(fake_states)
        fake_logits = F.log_softmax(fake_logits, dim=-1)

        # # get arc outputs
        offset_true = torch.zeros_like(input_ids)
        offset_true[:, :-1] = input_ids[:, 1:]
        offset_fake = torch.zeros_like(fake_ids)
        offset_fake[:, :-1] = fake_ids[:, 1:]

        ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
        true_lm_select = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_true.view(-1)]
        fake_lm_select = lm_logits.detach().view(-1, lm_logits.shape[-1])[ar, offset_fake.view(-1)]
        
        true_arc_select = true_logits.view(-1, true_logits.shape[-1])[ar, offset_true.view(-1)]
        fake_arc_select = fake_logits.view(-1, fake_logits.shape[-1])[ar, offset_fake.view(-1)]

        # negative because lower energy = more likely
        true_arc = -(true_arc_select - true_lm_select)
        fake_arc = -(fake_arc_select - fake_lm_select)

        true_arc = true_arc.view(batch_size, seq_length)
        fake_arc = fake_arc.view(batch_size, seq_length)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
