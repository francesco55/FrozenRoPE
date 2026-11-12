# Positional modules for posext: stock RoPE, PI, YaRN, learned frequencies, per-head
# temperatures (constant and AdaScale) and DaRoPE slow-band coordinates.
#
# Attribution:
# - `yarn_inv_freq` follows `_compute_yarn_parameters` in Hugging Face transformers 4.57.1
#   (src/transformers/modeling_rope_utils.py), Copyright 2024 The HuggingFace Team,
#   Apache License 2.0.
# - The AdaScale temperature follows `AdaScale` in AdaRoPE (github.com/Outsider565/AdaRoPE,
#   src/adarope/modules.py), Apache License 2.0. Same formula and initialization
#   (tau = 1, gamma = 0); we index heads per layer inside one model-level module.
# - DaRoPE (Levy et al., 2026) has no public code; the coordinate is implemented from the paper
#   as specified in CLAUDE.md.
"""Positional parameterizations for the patched OLMo-2 attention.

All positional parameters of the model live in one `PositionalEncoding` module owned by the
backbone (`model.model.posext`). Each attention layer calls it with its own `layer_idx`,
the `position_ids` of the batch and (for DaRoPE) its input hidden states, and gets back the
cos/sin it should rotate with and an optional per-head query scale.

Angles are always computed from `position_ids`, never from `arange(seq_len)`, so PoSE's
skipped positions work.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import torch
from torch import nn

# Allowed values for the string-valued config fields.
MODES = ("none", "pi", "yarn", "learned_freq", "learned_freq_temp", "darope")
GRANULARITIES = ("shared", "layer", "head")
TEMP_FORMS = ("constant", "adascale")
LENGTH_SOURCES = ("tokens", "positions")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def _from_dict(cls: type, d: dict[str, Any] | None) -> Any:
    """Build a dataclass from a dict, rejecting unknown keys so config typos fail loudly."""
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


@dataclass
class YarnConfig:
    """YaRN settings (Peng et al., 2024). Same semantics as transformers' `rope_scaling`."""

    factor: float = 4.0  # how much slow bands are interpolated (L' / L)
    beta_fast: float = 32.0  # bands rotating more than this many times in L are kept as-is
    beta_slow: float = 1.0  # bands rotating fewer than this many times in L are fully interpolated
    # None -> 0.1 * ln(factor) + 1, as in the paper and transformers.
    attention_factor: float | None = None
    truncate: bool = True  # round the ramp boundaries to whole bands (transformers default)


@dataclass
class TempConfig:
    """Per-head query temperature for `learned_freq_temp`."""

    form: str = "adascale"
    # What "length" means for AdaScale. Following AdaRoPE, the length is per query:
    # L_i = index_i + 1, where index_i is the token's index in the sequence (`tokens`,
    # from cache_position) or its position id (`positions`). They differ only under PoSE,
    # where `tokens` keeps every length <= L_ref, so gamma only duplicates tau as a constant
    # scale and learns nothing about length; RQ2 should run both. See CLAUDE.md,
    # "Positional modes" -> adascale.
    length_source: str = "tokens"
    # AdaScale reference length L_ref. None -> native length L.
    ref_len: int | None = None

    def __post_init__(self) -> None:
        """Validate the string fields."""
        if self.form not in TEMP_FORMS:
            raise ValueError(f"temp.form must be one of {TEMP_FORMS}, got {self.form!r}")
        if self.length_source not in LENGTH_SOURCES:
            raise ValueError(
                f"temp.length_source must be one of {LENGTH_SOURCES}, got {self.length_source!r}"
            )


@dataclass
class DaRoPEConfig:
    """DaRoPE slow-band coordinate settings."""

    # Clip for (m + 0.5) / L. Must be < 0.5 / L, otherwise positions inside L get clipped and
    # the model is no longer identical to stock at initialization.
    eps: float = 1e-5
    # Parameter-free RMS normalization of the attention input before projecting with w_h.
    rms_norm_input: bool = True


@dataclass
class PosConfig:
    """The `pos.*` config block. See CLAUDE.md, "Positional modes"."""

    mode: str = "none"
    granularity: str = "head"  # learned_freq / learned_freq_temp only
    pi_factor: float = 4.0  # pi only
    # Native (pretraining) context length L. None -> the model config's max_position_embeddings.
    native_len: int | None = None
    # Initialize learned frequencies at YaRN's values instead of the original ones.
    init_from_yarn: bool = False
    yarn: YarnConfig = field(default_factory=YarnConfig)
    temp: TempConfig = field(default_factory=TempConfig)
    darope: DaRoPEConfig = field(default_factory=DaRoPEConfig)

    def __post_init__(self) -> None:
        """Validate the string fields."""
        if self.mode not in MODES:
            raise ValueError(f"pos.mode must be one of {MODES}, got {self.mode!r}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"pos.granularity must be one of {GRANULARITIES}, got {self.granularity!r}"
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PosConfig:
        """Parse a (possibly nested, possibly partial) dict, e.g. the `pos:` block of a YAML."""
        d = dict(d or {})
        # Parse the nested blocks first so their unknown keys are reported with their own name.
        yarn = _from_dict(YarnConfig, d.pop("yarn", None))
        temp = _from_dict(TempConfig, d.pop("temp", None))
        darope = _from_dict(DaRoPEConfig, d.pop("darope", None))
        cfg = _from_dict(cls, {**d, "yarn": None, "temp": None, "darope": None})
        cfg.yarn, cfg.temp, cfg.darope = yarn, temp, darope
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (JSON-serializable), stored in the HF config as `posext`."""
        return asdict(self)

    @property
    def learns_freq(self) -> bool:
        """Whether this mode has learned frequency scales `a`."""
        return self.mode in ("learned_freq", "learned_freq_temp")


# --------------------------------------------------------------------------------------
# Frequencies
# --------------------------------------------------------------------------------------


def rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """Stock RoPE frequencies theta_j = base^(-2j / head_dim), j = 0 .. head_dim/2 - 1."""
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def slow_band_mask(inv_freq: torch.Tensor, native_len: int) -> torch.Tensor:
    """Bands whose wavelength 2*pi/theta_j exceeds the native length L."""
    return (2 * math.pi / inv_freq) > native_len


def yarn_attention_factor(cfg: YarnConfig) -> float:
    """YaRN's cos/sin multiplier (scales both q and k, so logits by its square)."""
    if cfg.attention_factor is not None:
        return float(cfg.attention_factor)
    return 1.0 if cfg.factor <= 1 else 0.1 * math.log(cfg.factor) + 1.0


def yarn_inv_freq(head_dim: int, base: float, native_len: int, cfg: YarnConfig) -> torch.Tensor:
    """YaRN frequencies: interpolate slow bands by `factor`, keep fast bands, ramp between.

    Mirrors transformers 4.57.1 `_compute_yarn_parameters` (Apache 2.0) so our `yarn` mode can
    be cross-checked against stock `rope_scaling={"rope_type": "yarn", ...}`.
    """
    dim = head_dim

    def correction_dim(num_rotations: float) -> float:
        """Band index (fractional) that completes `num_rotations` rotations within L."""
        return (dim * math.log(native_len / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Bands below `low` rotate fast (keep); bands above `high` rotate slowly (interpolate).
    low, high = correction_dim(cfg.beta_fast), correction_dim(cfg.beta_slow)
    if cfg.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001  # prevent singularity

    # Linear ramp from 0 (at `low`) to 1 (at `high`), then blend the two frequency sets.
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation_weight = 1 - ramp  # 1 on fast bands (keep), 0 on slow bands (interpolate)
    extrapolated = rope_inv_freq(head_dim, base)
    interpolated = extrapolated / cfg.factor
    return interpolated * (1 - extrapolation_weight) + extrapolated * extrapolation_weight


# --------------------------------------------------------------------------------------
# Rotation helpers
# --------------------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) on the two halves of the last dim (HF half-split layout)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q, k of shape (B, H, T, D) with cos/sin broadcastable to (B, H, T, D).

    Same arithmetic as transformers' `apply_rotary_pos_emb` (half-split layout, fp32 cos/sin,
    result cast back to the input dtype), so `none` mode reproduces the stock model exactly.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def _fp32(device: torch.device):
    """Context manager that disables autocast, so positional math runs in float32."""
    device_type = device.type if device.type in ("cuda", "cpu", "xpu") else "cpu"
    return torch.autocast(device_type=device_type, enabled=False)


# --------------------------------------------------------------------------------------
# The model-level positional module
# --------------------------------------------------------------------------------------


@dataclass
class RotaryOutput:
    """What one attention layer needs from `PositionalEncoding`."""

    cos: torch.Tensor  # (B, 1 or H, T, D), float32
    sin: torch.Tensor  # (B, 1 or H, T, D), float32
    q_scale: torch.Tensor | None  # (B or 1, H, T or 1, 1), float32, or None


class PositionalEncoding(nn.Module):
    """All positional parameters of the model, for every mode.

    Learned parameters (float32, regardless of the model dtype):
      learned_freq*: `a`        (1, 1, F) shared | (n_layers, 1, F) layer | (n_layers, H, F) head
      temp constant: `log_tau`  (n_layers, H)
      temp adascale: `tau`, `gamma` (n_layers, H)
      darope:        `w` (n_layers, H, hidden), `alpha` (n_layers, H)
    where F = head_dim / 2 is the number of rotary bands.
    """

    def __init__(
        self,
        pos: PosConfig,
        *,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        rope_base: float,
        native_len: int,
    ) -> None:
        """Create the buffers and the parameters needed by `pos.mode`, initialized to identity."""
        super().__init__()
        self.pos = pos
        self.n_layers, self.n_heads, self.head_dim = n_layers, n_heads, head_dim
        self.hidden_size = hidden_size
        self.rope_base, self.native_len = rope_base, native_len
        n_bands = head_dim // 2

        # Sanity checks on the configuration.
        per_head = pos.mode == "darope" or (pos.learns_freq and pos.granularity == "head")
        if per_head and n_kv_heads != n_heads:
            raise NotImplementedError("Per-head positional modes assume no GQA (n_kv_heads == n_heads).")
        if pos.mode == "darope" and not pos.darope.eps < 0.5 / native_len:
            raise ValueError(
                f"darope.eps={pos.darope.eps} must be < 0.5/L={0.5 / native_len:.3g} "
                "so that positions inside L are not clipped."
            )

        # Fixed buffers (non-persistent: recomputed from the config, never saved).
        self.register_buffer("inv_freq", rope_inv_freq(head_dim, rope_base), persistent=False)
        self.register_buffer("slow_mask", slow_band_mask(self.inv_freq, native_len), persistent=False)
        if pos.mode == "yarn" or pos.init_from_yarn:
            yarn = yarn_inv_freq(head_dim, rope_base, native_len, pos.yarn)
            self.register_buffer("yarn_inv_freq", yarn, persistent=False)

        # Learned parameters (only those the mode needs are created).
        if pos.learns_freq:
            # One row of band scales per layer (or one shared row); one column per head or 1.
            rows = {"shared": 1, "layer": n_layers, "head": n_layers}[pos.granularity]
            cols = n_heads if pos.granularity == "head" else 1
            self.a = nn.Parameter(torch.zeros(rows, cols, n_bands, dtype=torch.float32))
        if pos.mode == "learned_freq_temp":
            if pos.temp.form == "constant":
                self.log_tau = nn.Parameter(torch.zeros(n_layers, n_heads, dtype=torch.float32))
            else:
                self.tau = nn.Parameter(torch.ones(n_layers, n_heads, dtype=torch.float32))
                self.gamma = nn.Parameter(torch.zeros(n_layers, n_heads, dtype=torch.float32))
        if pos.mode == "darope":
            self.w = nn.Parameter(torch.zeros(n_layers, n_heads, hidden_size, dtype=torch.float32))
            self.alpha = nn.Parameter(torch.ones(n_layers, n_heads, dtype=torch.float32))
        self.reset_parameters()

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        """Keep every floating parameter and buffer in float32, even after `model.to(bfloat16)`.

        Learned positional parameters are finicky in low precision, and bf16 frequencies would
        give wrong angles at large positions. Device moves still apply normally.
        """
        # Keep the original float32 tensors: casting to bf16 and back would lose precision.
        saved_params = {n: p.data for n, p in self._parameters.items() if p is not None and p.dtype == torch.float32}
        saved_buffers = {n: b for n, b in self._buffers.items() if b is not None and b.dtype == torch.float32}
        super()._apply(fn, recurse)
        # Where fn changed the dtype, restore the original values on the new device.
        # `p.data = ...` keeps the same Parameter object (optimizer-safe).
        for name, original in saved_params.items():
            p = self._parameters[name]
            if p.dtype != torch.float32:
                p.data = original.to(device=p.device, dtype=torch.float32)
        for name, original in saved_buffers.items():
            b = self._buffers[name]
            if b.dtype != torch.float32:
                self._buffers[name] = original.to(device=b.device, dtype=torch.float32)
        return self

    @property
    def ref_len(self) -> int:
        """AdaScale reference length L_ref (defaults to the native length L)."""
        return self.pos.temp.ref_len or self.native_len

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize so that the model equals the stock model (or YaRN, if init_from_yarn).

        For `learned_freq` (no temperature parameters) init_from_yarn sets the frequencies only;
        YaRN's attention factor cannot be represented there.
        """
        if self.pos.learns_freq:
            if self.pos.init_from_yarn:
                # a = log(theta_yarn / theta), so exp(a) * theta reproduces YaRN's frequencies.
                self.a.copy_((self.yarn_inv_freq / self.inv_freq).log().expand_as(self.a))
            else:
                self.a.zero_()
        if self.pos.mode == "learned_freq_temp":
            # init_from_yarn also starts at YaRN's attention scaling, so the model equals `yarn`
            # mode at step 0: YaRN multiplies cos/sin by f (queries and keys), i.e. logits by f^2,
            # which a query-only scale of f^2 reproduces exactly. Otherwise the scale is 1 (stock).
            scale = yarn_attention_factor(self.pos.yarn) ** 2 if self.pos.init_from_yarn else 1.0
            if self.pos.temp.form == "constant":
                self.log_tau.fill_(math.log(scale))  # scale exp(log_tau) = f^2 (or 1)
            else:
                self.tau.fill_(1.0 / scale)  # with gamma = 0, lambda = 1 / tau at every length
                self.gamma.zero_()
        if self.pos.mode == "darope":
            self.w.zero_()  # no content term
            self.alpha.fill_(1.0)  # sigmoid(logit(p)) = p, so the coordinate is m + 0.5

    # ---- per-layer frequencies -------------------------------------------------------

    def layer_inv_freq(self, layer_idx: int) -> torch.Tensor:
        """Frequencies used by one layer, shape (1 or H, F)."""
        mode = self.pos.mode
        if mode == "yarn":
            return self.yarn_inv_freq[None]
        if self.pos.learns_freq:
            row = 0 if self.pos.granularity == "shared" else layer_idx
            return self.inv_freq[None] * self.a[row].float().exp()  # theta_hat = exp(a) * theta
        return self.inv_freq[None]  # none, pi, darope (fast bands)

    # ---- forward ----------------------------------------------------------------------

    def forward(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> RotaryOutput:
        """cos/sin (and optional query scale) for one layer.

        Args:
            layer_idx: Index of the calling attention layer.
            position_ids: (B, T) position ids of the current tokens.
            hidden_states: (B, T, hidden) attention input; required for DaRoPE.
            cache_position: (T,) token indices of the current tokens (for temp.length_source=tokens).
        """
        with _fp32(position_ids.device):
            pos = position_ids.float()
            if self.pos.mode == "pi":
                pos = pos / self.pos.pi_factor  # PI: compress positions into the trained range
            # angle[b, h, t, j] = position[b, t] * theta[h, j]  -> (B, 1 or H, T, F)
            angles = pos[:, None, :, None] * self.layer_inv_freq(layer_idx)[None, :, None, :]

            if self.pos.mode == "darope":
                angles = self._darope_angles(layer_idx, position_ids, hidden_states, angles)

            # Duplicate the F band angles to the full head_dim (HF half-split layout).
            emb = torch.cat((angles, angles), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            if self.pos.mode == "yarn":
                factor = yarn_attention_factor(self.pos.yarn)
                cos, sin = cos * factor, sin * factor

            q_scale = None
            if self.pos.mode == "learned_freq_temp":
                q_scale = self._query_scale(layer_idx, position_ids, cache_position)
        return RotaryOutput(cos=cos, sin=sin, q_scale=q_scale)

    def _darope_angles(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """Replace the position on slow bands with c_h(x_m) = L * sigmoid(w_h.x~_m + alpha_h * beta_m)."""
        if hidden_states is None:
            raise ValueError("darope mode needs the attention input hidden_states")
        L, eps = self.native_len, self.pos.darope.eps
        x = hidden_states.float()
        if self.pos.darope.rms_norm_input:
            # OLMo-2's attention input is the un-normalized residual stream; normalize its scale.
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # beta_m = logit((m + 0.5) / L), clipped so positions >= L saturate instead of exploding.
        frac = ((position_ids.float() + 0.5) / L).clamp(eps, 1 - eps)
        beta = torch.logit(frac)  # (B, T)
        # Content term w_h . x~_m for every head at once.
        content = torch.einsum("btd,hd->bht", x, self.w[layer_idx].float())  # (B, H, T)
        coord = L * torch.sigmoid(content + self.alpha[layer_idx].float()[None, :, None] * beta[:, None, :])
        slow = coord[..., None] * self.inv_freq  # (B, H, T, F); only slow bands are used
        fast = angles.expand(-1, self.n_heads, -1, -1)  # stock angles, broadcast per head
        return torch.where(self.slow_mask, slow, fast)  # pick per band

    def _query_scale(
        self, layer_idx: int, position_ids: torch.Tensor, cache_position: torch.Tensor | None
    ) -> torch.Tensor:
        """Per-head query multiplier for `learned_freq_temp` (constant or AdaScale)."""
        if self.pos.temp.form == "constant":
            return self.log_tau[layer_idx].float().exp()[None, :, None, None]  # (1, H, 1, 1)
        # AdaScale: lambda_h(L_i) = (1 / tau_h) * ln(1 + max(L_i, L_ref) / L_ref) ** gamma_h.
        if self.pos.temp.length_source == "positions":
            index = position_ids.float()  # (B, T)
        else:
            if cache_position is None:
                raise ValueError("temp.length_source='tokens' needs cache_position")
            index = cache_position.float()[None, :]  # (1, T)
        length = index[:, None, :, None] + 1  # L_i = keys seen by query i, (B or 1, 1, T, 1)
        growth = torch.log1p(length.clamp(min=self.ref_len) / self.ref_len)  # flat up to L_ref
        gamma = self.gamma[layer_idx].float()[None, :, None, None]
        tau = self.tau[layer_idx].float()[None, :, None, None]
        return growth.pow(gamma) / tau

    def extra_repr(self) -> str:
        """Summary shown when printing the model."""
        return (
            f"mode={self.pos.mode}, granularity={self.pos.granularity}, native_len={self.native_len}, "
            f"slow_bands={int(self.slow_mask.sum())}/{self.slow_mask.numel()}"
        )
