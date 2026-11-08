# Copied from Hugging Face transformers 4.57.1,
# src/transformers/models/olmo2/modeling_olmo2.py (generated from modular_olmo2.py).
# Copyright 2024 The HuggingFace Team. Licensed under the Apache License, Version 2.0.
#
# Changes for posext (search for "posext:"):
# - Relative imports made absolute, so the file also works as `trust_remote_code` custom code.
# - New `Olmo2PosExtConfig` with a `posext` dict (the `pos.*` config block).
# - The model-level rotary module is replaced by `PositionalEncoding` (positional.py), which
#   holds all positional parameters. Each attention layer computes its own cos/sin from its
#   own parameters, the `position_ids` and (for DaRoPE) its input hidden states, then applies
#   an optional per-head query temperature. QK-norm stays before the rotation, and the stock
#   attention backend (eager / SDPA / FlashAttention) runs after it.
# - Classes renamed Olmo2* -> Olmo2PosExt* where they changed.
# - Docstrings and brief comments added for readability.
"""OLMo-2 with configurable, learnable positional parameters (posext)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from .positional import PosConfig, PositionalEncoding, apply_rotary


# posext: config with the positional block.
class Olmo2PosExtConfig(Olmo2Config):
    """`Olmo2Config` plus `posext`, a dict matching `PosConfig` (the `pos.*` config block)."""

    model_type = "olmo2_posext"

    def __init__(self, posext: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        """Standard OLMo-2 config kwargs plus `posext` (validated and stored as a plain dict)."""
        super().__init__(**kwargs)
        # Round-trip through PosConfig to validate and fill defaults; a dict keeps it JSON-safe.
        self.posext = PosConfig.from_dict(posext).to_dict()

    @property
    def pos(self) -> PosConfig:
        """The positional block as a typed `PosConfig`."""
        return PosConfig.from_dict(self.posext)

    @property
    def resolved_head_dim(self) -> int:
        """Head dimension (OLMo-2 configs leave `head_dim` unset; derive it)."""
        return getattr(self, "head_dim", None) or self.hidden_size // self.num_attention_heads


@use_kernel_forward_from_hub("RMSNorm")
class Olmo2RMSNorm(nn.Module):
    """RMSNorm with a learned scale (unchanged from transformers)."""

    def __init__(self, hidden_size, eps=1e-6):
        """
        Olmo2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """Normalize by the root-mean-square in float32, scale, and cast back."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        """Summary shown when printing the model."""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """Plain PyTorch attention (softmax(q k^T * scaling + mask) v); used when attn is "eager"."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Olmo2PosExtAttention(nn.Module):
    """Multi-head attention with QK-norm, then our per-layer rotation and optional temperature."""

    def __init__(self, config: Olmo2PosExtConfig, layer_idx: Optional[int] = None):
        """Create the q/k/v/o projections and the QK-norms (weights load from the checkpoint)."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # QK-norm over the full projection (all heads together), applied before RoPE.
        self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        # posext: set by Olmo2PosExtModel.attach_positional(). Stored as a plain attribute, not a
        # submodule, because one PositionalEncoding is shared by all layers and is registered once
        # on the backbone (model.model.posext).
        object.__setattr__(self, "_posext", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project, QK-norm, rotate with this layer's positional params, then run attention."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Projections; QK-norm is applied before splitting into heads.
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        # (B, T, H*D) -> (B, H, T, D)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # posext: per-layer rotation from this layer's parameters and the position_ids,
        # applied after QK-norm exactly where stock RoPE is applied.
        if position_ids is None:
            raise ValueError("posext attention needs position_ids")
        rot = self._posext(self.layer_idx, position_ids, hidden_states, cache_position)
        cos, sin = rot.cos, rot.sin
        query_states, key_states = apply_rotary(query_states, key_states, cos, sin)
        if rot.q_scale is not None:
            # Per-head temperature: scale queries only; keys, values and the kernel are unchanged.
            query_states = (query_states * rot.q_scale).to(value_states.dtype)

        # Append the (already rotated) keys/values to the KV cache during generation.
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Pick the attention kernel (eager / sdpa / flash_attention_2 / flex) from the config.
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            position_ids=position_ids,
            **kwargs,
        )

        # (B, T, H, D) -> (B, T, H*D) -> output projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo2MLP(nn.Module):
    """SwiGLU feed-forward block (unchanged from transformers)."""

    def __init__(self, config):
        """Create the gate, up and down projections."""
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """down(act(gate(x)) * up(x))."""
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Olmo2PosExtDecoderLayer(GradientCheckpointingLayer):
    """One transformer block: attention and MLP, each followed by its norm (reordered norm)."""

    def __init__(self, config: Olmo2PosExtConfig, layer_idx: int):
        """Create the attention, MLP and the two post-norms."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo2PosExtAttention(config=config, layer_idx=layer_idx)

        self.mlp = Olmo2MLP(config)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """x + norm(attn(x)), then x + norm(mlp(x))."""
        # OLMo-2 uses reordered norm: attention sees the un-normalized residual stream. DaRoPE's
        # content projection therefore receives `hidden_states` here (normalized in positional.py).
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Olmo2PosExtPreTrainedModel(PreTrainedModel):
    """Shared HF plumbing: config class, loading rules, supported attention backends."""

    config_class = Olmo2PosExtConfig
    config: Olmo2PosExtConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Olmo2PosExtDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Olmo2PosExtDecoderLayer,
        "attentions": Olmo2PosExtAttention,
    }
    # posext: positional parameters are absent from stock OLMo-2 checkpoints; they are
    # initialized by PositionalEncoding.reset_parameters() (identity with the stock model).
    _keys_to_ignore_on_load_missing = [r"posext\."]
    def post_init(self) -> None:
        """HF post-init, then ask `from_pretrained` to keep positional params float32."""
        super().post_init()
        # posext: positional parameters stay float32 when loading with dtype=bfloat16. Set on the
        # instance, and only when posext has parameters: HF's post_init rejects names in this list
        # that own no parameters (modes none/pi/yarn only have buffers).
        if any(".posext." in f".{name}" for name, _ in self.named_parameters()):
            self._keep_in_fp32_modules_strict = ["posext"]

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize one module; positional params get their identity init, not random init."""
        # posext: never let the generic initializer touch positional parameters.
        if isinstance(module, PositionalEncoding):
            module.reset_parameters()
            return
        super()._init_weights(module)


class Olmo2PosExtModel(Olmo2PosExtPreTrainedModel):
    """The decoder stack (embeddings, layers, final norm) plus the shared positional module."""

    def __init__(self, config: Olmo2PosExtConfig):
        """Build the layers and the positional module, and link each attention layer to it."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo2PosExtDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # posext: replaces Olmo2RotaryEmbedding. One module holds all positional parameters.
        self.posext = self._build_positional(config)
        self.attach_positional()
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _build_positional(config: Olmo2PosExtConfig) -> PositionalEncoding:
        """Create the PositionalEncoding from the config, recording the native length L."""
        pos = config.pos
        if pos.native_len is None:
            # Record L in the config so saved/exported models keep it even if
            # max_position_embeddings is later raised to the extended length.
            pos.native_len = config.max_position_embeddings
            config.posext = pos.to_dict()
        if config.rope_scaling is not None:
            raise ValueError("Use posext.mode for scaling; config.rope_scaling must be None.")
        return PositionalEncoding(
            pos,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.resolved_head_dim,
            hidden_size=config.hidden_size,
            rope_base=config.rope_theta,
            native_len=pos.native_len,
        )

    def attach_positional(self) -> None:
        """Point every attention layer at the shared PositionalEncoding."""
        for layer in self.layers:
            # object.__setattr__ bypasses nn.Module registration (see Olmo2PosExtAttention).
            object.__setattr__(layer.self_attn, "_posext", self.posext)

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Embed tokens, run all decoder layers, apply the final norm."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # cache_position = token indices of the current tokens (continues after cached tokens).
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Default positions are the token indices; PoSE passes its own position_ids instead.
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        # posext: no shared cos/sin here; each attention layer computes its own rotation.

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Olmo2PosExtForCausalLM(Olmo2PosExtPreTrainedModel, GenerationMixin):
    """Decoder stack plus the language-model head; the class to load and train."""

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        """Build the backbone and the vocabulary projection."""
        super().__init__(config)
        self.model = Olmo2PosExtModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def posext(self) -> PositionalEncoding:
        """Shortcut to the positional module (`model.model.posext`)."""
        return self.model.posext

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Run the backbone, compute logits, and the next-token loss if `labels` are given."""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shifts labels internally: position t predicts token t + 1.
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Olmo2PosExtConfig",
    "Olmo2PosExtForCausalLM",
    "Olmo2PosExtModel",
    "Olmo2PosExtPreTrainedModel",
]
