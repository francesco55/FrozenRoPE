"""Export a trained posext run as a full HF model directory for RULER / lm-eval.

Writes `config.json` with `auto_map` pointing at our model classes, copies the model source
files into the directory, and saves the full weights (backbone + positional). Callers must
pass `trust_remote_code=True` when loading.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from posext.models.build import load_trained
from posext.models.olmo2_posext import Olmo2PosExtConfig, Olmo2PosExtForCausalLM, Olmo2PosExtModel

logger = logging.getLogger(__name__)

_MODEL_FILES = ("olmo2_posext.py", "positional.py")
_PKG_MODELS = Path(__file__).resolve().parent / "models"


def export_full_model(run_dir: str | Path, export_dir: str | Path) -> Path:
    """Materialize a self-contained HF directory from a `save_positional` run."""
    run_dir, export_dir = Path(run_dir), Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # bf16 on CPU keeps the export compact; load_trained defaults to bf16.
    model = load_trained(run_dir)

    config: Olmo2PosExtConfig = model.config
    config.architectures = ["Olmo2PosExtForCausalLM"]
    config.auto_map = {
        "AutoConfig": "olmo2_posext.Olmo2PosExtConfig",
        "AutoModel": "olmo2_posext.Olmo2PosExtModel",
        "AutoModelForCausalLM": "olmo2_posext.Olmo2PosExtForCausalLM",
    }
    config.model_type = "olmo2_posext"
    config.save_pretrained(export_dir)
    model.save_pretrained(export_dir)

    for name in _MODEL_FILES:
        shutil.copy2(_PKG_MODELS / name, export_dir / name)

    # Tiny loader stub so remote code can import sibling modules if needed.
    (export_dir / "__init__.py").write_text(
        "from .olmo2_posext import Olmo2PosExtConfig, Olmo2PosExtForCausalLM, Olmo2PosExtModel\n"
    )
    meta = {
        "source_run": str(run_dir.resolve()),
        "classes": {
            "config": Olmo2PosExtConfig.__name__,
            "model": Olmo2PosExtModel.__name__,
            "causal_lm": Olmo2PosExtForCausalLM.__name__,
        },
    }
    (export_dir / "posext_export.json").write_text(json.dumps(meta, indent=2))
    logger.info("Exported full model to %s", export_dir)
    return export_dir
