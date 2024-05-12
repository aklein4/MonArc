from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools
try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
except ImportError:
    print("WARNING: torch_xla not installed, cannot use gradient checkpointing")

from transformers.modeling_utils import PreTrainedModel
from transformers.models.stablelm.modeling_stablelm import StableLmDecoderLayer

from models.arc.configuration_arc import ArcConfig
from utils.data_utils import DotDict
from utils.logging_utils import log_print


class ArcPreTrainedModel(PreTrainedModel):

    # all StableLM/renamed
    config_class = ArcConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StableLmDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_sdpa = True

    # from StableLM
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    # converted from torch to torch xla
    def xla_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        gradient_checkpointing_func = functools.partial(xla_checkpoint_fn, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


    def _xla_set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        """ Set gradient checkpointing for base model and submodules. """
        
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )


class ArcModel(ArcPreTrainedModel):

    def __init__(self, config: ArcConfig):
        """
        StableLM-based language model.

        Args:
            config: ArcConfig
        """
        super().__init__(config)

        # vocab info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Standard weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False # found by _xla_set_gradient_checkpointing
        if config._gradient_checkpointing:
            log_print("Gradient checkpointing enabled!")
            self.xla_gradient_checkpointing_enable()

        # Initialize weights and apply final processing
        self.post_init()


    def _get_tokens(
        self,
        input_ids: torch.LongTensor
    ) -> torch.Tensor:
        """ Get the tokens that serve as transformer inputs.

        Args:
            input_ids (torch.LongTensor): Input token ids [bs, seq_length]
    
        Returns:
            torch.Tensor: inputs to transformer [bs, seq_length, hidden_size]
        """
        return self.embed_tokens(input_ids)

    
    @torch.no_grad()
    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor]=None
    ) -> torch.BoolTensor:
        """ Get the attention mask for the transformer.
         - True = MASKED
         - autoreregessive mask if not provided

        Args:
            input_ids (torch.LongTensor): Input token ids [bs, seq_length]
            mask (Optional[torch.BoolTensor], optional): Attention mask [bs/None, seq_length, seq_length]. True = MASKED. Defaults to None.

        Returns:
            torch.BoolTensor: sdpa attention mask [bs|1, 1, seq_length, seq_length]
        """
        batch_size, seq_length = input_ids.shape

        # default mask
        if mask is None:
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
        
        # process for attn version
        if self._attn_implementation == 'eager':
            # eager uses attn bias
            # https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/stablelm/modeling_stablelm.py#L290
            mask = torch.masked_fill(torch.zeros_like(mask).float(), mask, float('-inf'))
        elif self._attn_implementation == 'sdpa':
            # sdpa uses True = NOT masked
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            mask = ~mask
        else:
            # no mask for flash attention
            mask = None

        # final processing
        if mask is not None:

            # must have batch dimension
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            # cannot broadcast to batch size
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1)

            # head dim
            mask = mask.unsqueeze(1)

        return mask


    @torch.no_grad()
    def _get_position_ids(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None
    ) -> torch.LongTensor:
        """ Get the position ids for the transformer
         - sequential if not provided

        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            
        Returns:
            torch.LongTensor: position ids [1, seq_length]
        """
        batch_size, seq_length = input_ids.shape
        
        # default
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)

        # must have batch dimension
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        return position_ids


    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
    ) -> torch.Tensor:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.

        Returns:
            torch.Tensor: final hidden states [bs, seq_length, hidden_size]
        """
        batch_size, seq_length = input_ids.shape

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                )[0]

            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]

        return self.norm(hidden_states)


class ArcLMModel(ArcPreTrainedModel):

    def __init__(self, config: ArcConfig):
        """ Arc model with a linear head for language modeling.
         - uses linear head for arc modeling
        
        Args:
            config (AnnelidConfig): Annelid configuration
        """
        super().__init__(config)

        # transformer
        self.model = ArcModel(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.arc_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    @torch.no_grad()
    def _get_arc_mask(
        self,
        input_ids: torch.LongTensor
    ) -> torch.BoolTensor:
        """ Get the mask for arc prediction.
         - True = MASKED

        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length]

        Returns:
            torch.BoolTensor: arc mask [seq_length, seq_length]
        """
        batch_size, seq_length = input_ids.shape

        # mask with no self-attention
        full_mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)

        # self attending
        nw = torch.triu(full_mask, diagonal=1)

        # cross attending
        sw = torch.triu(full_mask, diagonal=0)

        # empty
        ne = full_mask

        # only self
        se = ~torch.eye(seq_length, dtype=torch.bool, device=input_ids.device)

        # combine
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
        input_ids: torch.LongTensor
    ) -> torch.LongTensor:
        """ Get the position ids for arc prediction.

        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length]

        Returns:
            torch.LongTensor: arc position ids [1, seq_length]
        """
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, dtype=input_ids.dtype, device=input_ids.device)

        return torch.cat(
            [position_ids, position_ids],
            dim=0
        )


    def forward(self, input_ids):
        out = self.model(
            input_ids,
        )

        # get lm predictions
        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return DotDict(
            lm_logits=lm_logits
        )


    def train_forward(
        self,
        input_ids: torch.LongTensor,
        pad_token_id: int,
        debug: Optional[bool] = False
    ):
        """ Forward pass of the LM for training. 
         - creates negative samples
         - returns lm logits and arc predictions
         - 1 in arc predictions is positive, 0 is negative, -1 is padding
         
        Args:
            input_ids (torch.LongTensor): input token ids [bs, seq_length].
            pad_token_id (int): id of the pad token in the vocabulary.
            debug (Optional[bool], optional): Debug mode. Defaults to False.

        Returns:
            torch.Tensor: log-softmaxed logits [bs, seq_length, vocab_size]
            torch.Tensor: arc predictions [bs, seq_length-2]
            torch.Tensor: arc targets [bs, seq_length-2]
        """
        batch_size, seq_length = input_ids.shape

        # sample negative examples
        with torch.no_grad():

            # og_state = self.model.training
            # self.model.eval()
            out_sample = self.model(input_ids)
            # self.model.train(og_state)

            logits_sample = self.lm_head(out_sample)
            logits_sample = F.log_softmax(logits_sample, dim=-1)
            dist = torch.distributions.Categorical(logits=logits_sample)
            
            neg_ids = dist.sample()
            if debug:
                neg_ids = input_ids.clone()
                neg_ids[:, :-1] = input_ids[:, 1:]

            arc_ids = torch.cat(
                [
                    input_ids,
                    input_ids[:, :1],
                    neg_ids[:, :-1]
                ],
                dim=1
            )

        # get arc inputs
        arc_mask = self._get_arc_mask(input_ids)
        arc_position_ids = self._get_arc_position_ids(input_ids)

        # run transformer
        out = self.model(
            input_ids=arc_ids,
            position_ids=arc_position_ids,
            attention_mask=arc_mask
        )

        # get lm predictions
        lm_logits = self.lm_head(out[:, :seq_length])
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        # get arc predictions
        # formated to use cross entropy loss
        arc_preds = self.arc_head(out)[:, :, 0]
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
