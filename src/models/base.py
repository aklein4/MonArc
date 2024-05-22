from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# constants handles import error warning
try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
except ImportError:
    pass

import functools

from transformers.modeling_utils import PreTrainedModel
from transformers.models.stablelm.modeling_stablelm import StableLmDecoderLayer
from transformers.models.stablelm.configuration_stablelm import StableLmConfig
from transformers.cache_utils import Cache

from utils.data_utils import DotDict
from utils.logging_utils import log_print
import utils.constants as constants


class BaseConfig(StableLmConfig):

    model_type = "base"

    def __init__(
        self,
        *args,
        gradient_checkpointing_layers: int=1_000_000,
        **kwargs
    ):

        # custom args
        self.gradient_checkpointing_layers = gradient_checkpointing_layers

        # backward compatibility
        kwargs["use_parallel_residual"] = kwargs.get("use_parallel_residual", False)

        # init with work arounds
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing


class BaseModel(PreTrainedModel):

    # all StableLM/renamed
    config_class = BaseConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["StableLmDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_sdpa = True

    # init with work arounds
    def __init__(self, config: BaseConfig):
        tmp_attn_implementation = config._attn_implementation
        config._attn_implementation = 'eager'
        super().__init__(config)
        config._attn_implementation = tmp_attn_implementation


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
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        
        gradient_checkpointing_func = functools.partial(xla_checkpoint_fn, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        
        log_print("Gradient checkpointing enabled!")


class BaseTransformer(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [StableLmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing_layers = config.gradient_checkpointing_layers
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def _get_tokens(
        self,
        input_ids: torch.LongTensor
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    
    @torch.no_grad()
    def _get_mask(
        self,
        input_ids: torch.LongTensor,
        mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        if cached_mask:
            return mask

        # error check
        if (mask is not None or segment_ids is not None) and self._attn_implementation.count('flash_attention_2'):
            raise ValueError("Custom attention mask and segmend_ids are not supported for Flash Attention!")

        # default eager causal mask
        if mask is None:
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            assert mask.dtype == torch.bool, f"Non-cached mask must be boolean, got {mask.dtype}"

        # must have batch dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # apply segment ids
        if segment_ids is not None:
            assert segment_ids.shape == input_ids.shape, f"Segment ids ({segment_ids.shape}) must have same shape as input ids ({input_ids.shape})"

            segment_mask = segment_ids[:, None, :] != segment_ids[:, :, None]
            mask = mask | segment_mask

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
            mask = None

        # final processing
        if mask is not None:

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
        batch_size, seq_length = input_ids.shape
        
        # default
        # we use relative position ids so segment_ids can be ignored
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
        segment_ids: Optional[torch.LongTensor]=None,
        cached_mask=False,
        kv: Optional[Cache]=None,
        disable_norm=False,
    ) -> DotDict:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.
            kv (Optional[Cache], optional): Key-Value cache. Defaults to None.
            
        Returns:
            DotDict:
                hidden_states [bs, seq_length, hidden_size]
                kv [Cache]
        """
        batch_size, seq_length = input_ids.shape

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask, segment_ids, cached_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        for layer_idx, decoder_layer in enumerate(self.layers):

            if self.gradient_checkpointing and self.training and layer_idx < self.gradient_checkpointing_layers:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

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
                    past_key_value=kv,
                    output_attentions=False,
                    use_cache=(kv is not None),
                )[0]

        if disable_norm:
            return hidden_states
        return self.norm(hidden_states)


class BaseLmModel(BaseModel):

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        # transformer
        self.model = BaseTransformer(config)

        # lm modeling
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.BoolTensor]=None,
        segment_ids: Optional[torch.LongTensor]=None,
        kv: Optional[Cache]=None,
    ) -> DotDict:
        """ Forward pass of the LM

        Args:
            input_ids (torch.LongTensor): token input ids [bs, seq_length]
            position_ids (Optional[torch.LongTensor], optional): Position ids [bs|None, seq_length]. Defaults to None.
            attention_mask (Optional[torch.BoolTensor], optional): Attention mask [bs|None, seq_length, seq_length]. True = MASKED. Defaults to None.
            kv (Optional[Cache], optional): Key-Value cache. Defaults to None.
        
        Returns:
            DotDict:
                lm_logits: log-softmaxed token probs [bs, seq_length, vocab_size]
        """

        # get lm predictions
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            kv=kv
        )

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
