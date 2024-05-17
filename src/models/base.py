from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint_fn
    _xla_found = True
except ImportError:
    _xla_found = False # constants handles import errors
try:
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attn_xla
except ImportError:
    if _xla_found:
        print("WARNING: flash_attention not found for torch_xla", flush=True)

import functools

from transformers.modeling_utils import PreTrainedModel
from transformers.models.stablelm.configuration_stablelm import StableLmConfig
from transformers.models.stablelm.modeling_stablelm import (
    StableLmAttention,
    StableLmSdpaAttention,
    StableLmFlashAttention2,
    StableLmMLP,
    StableLmDecoderLayer
)
from transformers.cache_utils import Cache

from utils.data_utils import DotDict
from utils.logging_utils import log_print
import utils.constants as constants


class BaseConfig(StableLmConfig):

    model_type = "base"

    def __init__(self, *args, **kwargs):

        # backward compatibility
        kwargs["use_parallel_residual"] = kwargs.get("use_parallel_residual", False)
        
        # init with work arounds
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        super().__init__(**kwargs)
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


class StableLmFlashAttention2XLA(StableLmFlashAttention2):

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):

        # We won't support attention mask
        if attention_mask is not None:
            raise ValueError("Attention mask is not supported for Flash Attention 2 on XLA!")

        return flash_attn_xla(
            query_states,  # [batch_size, num_heads, q_seq_len, d_model]
            key_states,  # [batch_size, num_heads, kv_seq_len, d_model]
            value_states,  # [batch_size, num_heads, kv_seq_len, d_model]
            causal=True # only support causal
        )


ATTENTION_CLASSES = {
    "eager": StableLmAttention,
    "sdpa": StableLmSdpaAttention,
    "flash_attention_2": StableLmFlashAttention2,
    "flash_attention_2_xla": StableLmFlashAttention2XLA,
}


class BaseDecoderLayer(StableLmDecoderLayer):

    def __init__(self, config: StableLmConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.use_parallel_residual = config.use_parallel_residual
        self.hidden_size = config.hidden_size
        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.mlp = StableLmMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = None
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)


class BaseTransformer(BaseModel):

    def __init__(self, config: BaseConfig, disable_norm=False):
        super().__init__(config)

        # info
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.disable_norm = disable_norm

        # weights
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BaseDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # optionally disable norm
        self.norm = nn.Identity() if disable_norm else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Compute configuration
        self._attn_implementation = config._attn_implementation

        # training configuration
        self.gradient_checkpointing = False # found by _xla_set_gradient_checkpointing
        if config.gradient_checkpointing:
            log_print("Gradient checkpointing enabled!")
            self.xla_gradient_checkpointing_enable()

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
        mask: Optional[torch.BoolTensor]=None
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        # default eager causal mask
        if mask is None and self._attn_implementation == 'eager':
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

        # check for custom mask
        if mask is not None and self._attn_implementation.count('flash_attention_2'):
            raise ValueError("Custom attention mask is not supported for Flash Attention 2!")

        # process for attn version
        if self._attn_implementation == 'eager':
            # eager uses attn bias
            # https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/stablelm/modeling_stablelm.py#L290
            mask = torch.masked_fill(torch.zeros_like(mask).float(), mask, float('-inf'))
        elif mask is not None and self._attn_implementation == 'sdpa':
            # sdpa uses True = NOT masked
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            mask = ~mask

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
                hidden_states [bs, seq_length, hidden_size]
                kv [Cache]
        """
        batch_size, seq_length = input_ids.shape

        # get inputs
        hidden_states = self._get_tokens(input_ids)
        attention_mask = self._get_mask(input_ids, attention_mask)
        position_ids = self._get_position_ids(input_ids, position_ids)

        # run transformer
        for decoder_layer in self.layers:

            if self.gradient_checkpointing and self.training:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

                print(self._gradient_checkpointing_func.__name__)
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
            kv=kv
        )

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
