"""Thin wrapper around NVIDIA RULER (subset of tasks at 4k/8k/16k).

RULER is a separate codebase; this module documents the expected interface and fails
loudly with setup instructions until the repo is cloned beside this project.

Expected layout (configurable via --ruler-root):

    third_party/RULER/   # git clone https://github.com/NVIDIA/RULER.git

RQ1 success criterion uses 16k RULER + passkey. Until the wrapper is wired, run RULER
manually against an exported full model (`python -m posext.eval.shortctx` exports one).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from posext.export import export_full_model

logger = logging.getLogger("posext.eval.ruler")

# Subset named in the proposal.
DEFAULT_TASKS = ("niah_single_1", "niah_multikey_1", "variable_tracking")
DEFAULT_LENGTHS = (4096, 8192, 16384)


def evaluate_ruler(
    model_path: str | Path,
    *,
    ruler_root: str | Path = "third_party/RULER",
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
) -> dict[str, Any]:
    """Run RULER if the vendor tree is present; otherwise raise with setup steps."""
    root = Path(ruler_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"RULER not found at {root}. Clone it first:\n"
            f"  git clone https://github.com/NVIDIA/RULER.git {root}\n"
            "Then install its dependencies and re-run. Until this wrapper is finished, "
            "export the model with `export_full_model` and follow RULER's own scripts."
        )
    # TODO: shell out to RULER's scripts with model_path, tasks, lengths; parse JSON scores.
    raise NotImplementedError(
        "RULER subprocess wiring is still a stub. "
        f"Model ready at {model_path}; tasks={tasks}; lengths={lengths}; root={root}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Trained positional run dir")
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--ruler-root", type=Path, default=Path("third_party/RULER"))
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--lengths", default="4096,8192,16384")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: export then attempt RULER."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    export_dir = args.export_dir or (args.run / "full_model")
    model_path = export_full_model(args.run, export_dir)
    tasks = tuple(t for t in args.tasks.split(",") if t)
    lengths = tuple(int(x) for x in args.lengths.split(",") if x)
    try:
        results = evaluate_ruler(model_path, ruler_root=args.ruler_root, tasks=tasks, lengths=lengths)
    except (FileNotFoundError, NotImplementedError) as exc:
        results = {"status": "stub", "error": str(exc), "model": str(model_path)}
        logger.error("%s", exc)
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
