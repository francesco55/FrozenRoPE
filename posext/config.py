"""Load and validate experiment YAML configs for training and evaluation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from posext.models.build import DEFAULT_MODEL
from posext.models.positional import PosConfig


def _from_dict(cls: type, d: dict[str, Any] | None) -> Any:
    """Build a dataclass from a dict, rejecting unknown keys so config typos fail loudly."""
    d = dict(d or {})
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class DataConfig:
    """PG-19 / PoSE data settings."""

    regime: str = "long"  # long | pose
    split: str = "train"
    seq_len: int = 16384  # token length of each chunk (4096 under pose)
    target_len: int = 16384  # L' for position ids
    max_sequences: int | None = 100
    max_chunks_per_book: int | None = 1
    n_chunks: int = 2  # pose only
    num_workers: int = 0

    def __post_init__(self) -> None:
        """Validate regime and lengths."""
        if self.regime not in ("long", "pose"):
            raise ValueError(f"data.regime must be 'long' or 'pose', got {self.regime!r}")
        if self.seq_len < 2 or self.target_len < self.seq_len:
            raise ValueError(
                f"need 2 <= seq_len <= target_len, got seq_len={self.seq_len}, target_len={self.target_len}"
            )


@dataclass
class TrainHyperParams:
    """Optimizer / loop settings for positional-only training."""

    steps: int = 500
    batch_size: int = 1
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # global clip; ignored for a group if a per-group clip is set
    clip_freq: float | None = None  # clip only frequency-scale params (a)
    clip_temp: float | None = None  # clip only temperature params
    logit_chunk_size: int = 2048  # vocab CE in chunks to fit 16k in memory
    log_every: int = 10
    save_every: int = 100
    seed: int = 0


@dataclass
class WandbConfig:
    """Optional Weights & Biases logging."""

    enabled: bool = True
    project: str = "frozenrope"
    entity: str | None = None
    run_name: str | None = None


@dataclass
class ExperimentConfig:
    """One reproducible experiment (YAML file)."""

    name: str
    pos: dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainHyperParams = field(default_factory=TrainHyperParams)
    model: str = DEFAULT_MODEL
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    output_dir: str = "outputs/run"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int | None = None  # if set, overrides train.seed

    def __post_init__(self) -> None:
        """Normalize nested configs and seeds."""
        if isinstance(self.data, dict):
            self.data = _from_dict(DataConfig, self.data)
        if isinstance(self.train, dict):
            t = dict(self.train)
            if "betas" in t and isinstance(t["betas"], list):
                t["betas"] = tuple(t["betas"])
            self.train = _from_dict(TrainHyperParams, t)
        if isinstance(self.wandb, dict):
            self.wandb = _from_dict(WandbConfig, self.wandb)
        if self.seed is not None:
            self.train.seed = self.seed
        # Validate pos early so typos fail before downloading weights.
        PosConfig.from_dict(self.pos)

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serializable dict (stores with results)."""
        return {
            "name": self.name,
            "model": self.model,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "output_dir": self.output_dir,
            "pos": dict(self.pos),
            "data": asdict(self.data),
            "train": asdict(self.train),
            "wandb": asdict(self.wandb),
            "seed": self.train.seed,
        }


def config_from_dict(raw: dict[str, Any], *, default_name: str | None = None) -> ExperimentConfig:
    """Validate a raw config mapping (as loaded from YAML); unknown keys are rejected."""
    raw = copy.deepcopy(raw)
    if "name" not in raw and default_name is not None:
        raw["name"] = default_name
    return _from_dict(ExperimentConfig, raw)


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """The YAML mapping of an experiment config, unvalidated."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(raw)}")  # noqa: TRY004
    return raw


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML; unknown top-level keys are rejected."""
    path = Path(path)
    return config_from_dict(load_raw_config(path), default_name=path.stem)
