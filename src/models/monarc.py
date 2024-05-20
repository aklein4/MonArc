from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from transformers.models.stablelm.modeling_stablelm import (
    StableLmAttention,
    StableLmMLP,
    apply_rotary_pos_emb,
    repeat_kv
)

from models.base import (
    BaseConfig,
    BaseModel,
    BaseTransformer,
)
from models.arc import ArcDecoderLayer
from utils.data_utils import DotDict
from utils.logging_utils import log_print


class MonArcConfig(BaseConfig):

    model_type = "monarc"

    def __init__(
        self,
        *args,
        num_head_layers=4,
        control=False,
        head_gradient_checkpointing=False,
        **kwargs
    ):
        self.num_head_layers = num_head_layers
        self.control = control
        self.head_gradient_checkpointing = head_gradient_checkpointing

        super().__init__(*args, **kwargs)


class MonArcHeadTransformer(BaseTransformer):

    def __init__(self, config: MonArcConfig):
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ArcDecoderLayer(config, layer_idx) for layer_idx in range(config.num_head_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.head_gradient_checkpointing = config.head_gradient_checkpointing
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.LongTensor]=None,
        memory: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv=None,
    ) -> DotDict:

        # get inputs
        if input_ids is not None:
            hidden_states = hidden_states + self._get_tokens(input_ids)
        else:
            input_ids = torch.zeros(hidden_states.shape[:-1], dtype=torch.long, device=hidden_states.device)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        mem_out = []
        for layer_idx, decoder_layer in enumerate(self.layers):

            mem_out.append(hidden_states)
            if memory is None:
                mem_in = None
            else:
                mem_in = memory[layer_idx]

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

        return self.norm(hidden_states), torch.stack(mem_out, dim=0)


class MonArcLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config, disable_norm=True)
        self.head_model = MonArcHeadTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # arc modeling
        self.arc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # fast sampling info
        self.vocab_factor = int(np.round(np.sqrt(self.vocab_size)))
        while self.vocab_size % self.vocab_factor != 0:
            self.vocab_factor += 1
        self.vocab_chunk = self.vocab_size // self.vocab_factor

        # extras
        self.control = config.control
        if self.control:
            log_print("MonArc control mode enabled!")

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

        # reuse the attention mask
        attention_mask = self.model._get_mask(input_ids, None, segment_ids)

        # get transformer output
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            cached_mask=True
        )

        # get the lm logits
        lm_states, memory = self.head_model(
            hidden_states,
            memory=None,
            attention_mask=attention_mask,
            cached_mask=True
        )
        lm_logits = self.lm_head(lm_states)

        # get the true labels
        # the last token is discarded later in the loss
        true_labels = input_ids.clone()
        true_labels[:, :-1] = input_ids[:, 1:]

        # get the fake labels
        fake_labels = input_ids.clone()

        factored_probs = torch.softmax(
            lm_logits.detach().float(), dim=-1
        ).view(-1, self.vocab_factor, self.vocab_chunk)

        outer_probs = factored_probs.sum(dim=-1)
        outer_sample = torch.multinomial(outer_probs, 1, True)[:, 0]

        ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
        inner_probs = factored_probs[ar, outer_sample]
        inner_sample = torch.multinomial(inner_probs, 1, True)[:, 0]

        sample = (self.vocab_chunk*outer_sample + inner_sample).view(batch_size, seq_length)
        fake_labels[:, :-1] = sample[:, :-1]

        # get the input tokens for the head
        if debug:
            true_tokens = None
            fake_tokens = None
        elif self.control:
            true_tokens = torch.zeros_like(true_labels)
            fake_tokens = torch.zeros_like(fake_labels)
        else:
            true_tokens = true_labels
            fake_tokens = fake_labels

        # get the true fake head outputs
        true_states, fake_states = self.head_model(
            torch.cat([hidden_states, hidden_states], dim=0),
            input_ids=(None if debug else torch.cat([true_tokens, fake_tokens], dim=0)),
            memory=torch.cat([memory, memory], dim=1),
            attention_mask=torch.cat([attention_mask, attention_mask], dim=0),
            cached_mask=True
        )[0].chunk(2, dim=0)

        # get the true and fake logits
        true_arc = torch.bmm(
            self.arc_head.weight[true_labels.view(-1)].unsqueeze(-2),
            true_states.view(-1, true_states.shape[-1]).unsqueeze(-1)
        )[:, 0, 0].reshape(batch_size, seq_length)
        fake_arc = torch.bmm(
            self.arc_head.weight[fake_labels.view(-1)].unsqueeze(-2),
            fake_states.view(-1, fake_states.shape[-1]).unsqueeze(-1)
        )[:, 0, 0].reshape(batch_size, seq_length)

        # # get arc outputs
        # ar = torch.arange(batch_size*seq_length, device=input_ids.device, dtype=torch.long)
        # tmp_lm_logits = lm_logits.view(-1, lm_logits.shape[-1]).detach()

        # true_arc = true_logits - tmp_lm_logits[ar, true_labels.view(-1)]
        # fake_arc = fake_logits - tmp_lm_logits[ar, fake_labels.view(-1)]

        # # flip sign so higher = lower residual = more likely
        # true_arc = -true_arc.view(batch_size, seq_length)
        # fake_arc = -fake_arc.view(batch_size, seq_length)

        # final processing
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return (
            lm_logits,
            true_arc,
            fake_arc
        )
