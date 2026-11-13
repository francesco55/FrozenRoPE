"""One-command SCC pipeline: hyperparameter sweep -> validation selection -> test evaluation.

On an SCC login node (the wrapper sources env.sh and cds to the repo):

    bash scripts/scc/pipeline.sh configs/rq1/learned_freq_head.yaml --sweep train.lr=1e-3,1e-2,3e-2
    bash scripts/scc/pipeline.sh configs/rq1/learned_freq_temp_head.yaml \\
        --sweep train.lr=1e-2,3e-2 --sweep pos.temp.form=adascale,constant --name temp_head_v1
    bash scripts/scc/pipeline.sh ... --dry-run          # print the plan and qsub commands only
    uv run python -m posext.pipeline status outputs/sweeps/<name>   # next morning
    # later: re-run test passkey on every sweep's selected model with more prompts
    bash scripts/scc/pipeline.sh retest outputs/sweeps/* --passkey-n-per-cell 50   # standard (v2) prompt

    # training-free baselines (none / pi / yarn) use the same call: no training, same evals
    bash scripts/scc/pipeline.sh configs/rq1/yarn_tuned.yaml

One call queues everything, chained with `-hold_jid`:

  1. `train` array job (GPU, one task per variant): train (skipped for training-free modes)
     -> PG-19 validation PPL -> validation passkey (seed 0). Skips what it already finished.
  2. `ruler_prepare` (CPU, test split), only if RULER test data is missing; runs alongside 1.
  3. `select` (CPU): rank variants on validation, write `selection.json`.
  4. `test` (GPU): PG-19 test PPL + passkey (seed 1) for the selected variant only.
  5. `ruler` array job (GPU, one task per length): RULER test split (500 samples per task) for
     the selected variant, then `ruler_merge` (CPU) combines the lengths into one result.

Selection uses validation data only (PG-19 validation split, passkey seed 0); the test split
and passkey seed 1 are touched only by the winner. Default score is the YaRN grid's combined
z-score (NLL at the longest length + passkey accuracy), so learned runs and the tuned YaRN
baseline are chosen the same way. Sweep hyperparameters, not seeds: picking the best seed on
validation is not a result.

Layout: `outputs/sweeps/<name>/` holds `plan.json`, generated `configs/`, `runs/<variant>/`
(positional params + `val_ppl.json` / `val_passkey.json`), `selection.json`, `logs/` and
`ruler_test/` predictions (+ per-length partial results). Test results go to `results/sweeps/<name>/test_{ppl,passkey,ruler}_<variant>.json`.

Re-running the same command with the same `--name` is safe: it resubmits, and every stage
skips work whose outputs exist (training resumes from `train_state.pt`, RULER from its
prediction files). A different plan under an existing name is refused.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from posext.config import config_from_dict, load_raw_config

logger = logging.getLogger("posext.pipeline")

SWEEP_ROOT = Path("outputs/sweeps")
RESULTS_ROOT = Path("results/sweeps")
STAGE_SCRIPT = Path("scripts/scc/sweep_stage.qsub")
PREPARE_SCRIPT = Path("scripts/scc/ruler_prepare.qsub")

# Same request as scripts/scc/train.qsub: Ampere+ (bf16), >= 40 GB. h_rt is added per stage.
GPU_RESOURCES = ("-l", "gpus=1", "-l", "gpu_c=8.0", "-l", "gpu_memory=40G", "-pe", "omp", "8")
CPU_RESOURCES = ("-pe", "omp", "1")
# Environment forwarded to jobs (SGE does not forward the submitting shell's environment).
FORWARD_ENV = ("POSEXT_ROOT", "POSEXT_STUDENT_ROOT", "POSEXT_DATA_CACHE", "POSEXT_RULER_ROOT")
TRAINING_FREE_MODES = ("none", "pi", "yarn")
METRICS = ("combined", "ppl", "passkey")

VAL_PASSKEY_SEED = 0  # validation prompts (same as the YaRN grid and val_ppl.qsub)
TEST_PASSKEY_SEED = 1  # test prompts (same as eval.qsub)


# ----------------------------------------------------------------------------------------
# Building the plan.


def parse_value(raw: str) -> Any:
    """A sweep value from the command line: YAML scalar, with `1e-3`-style floats accepted."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_sweep(spec: str) -> tuple[str, list[str]]:
    """`train.lr=1e-3,3e-2` -> ("train.lr", ["1e-3", "3e-2"]) (values kept raw for naming)."""
    key, sep, values = spec.partition("=")
    key = key.strip()
    raws = [v.strip() for v in values.split(",") if v.strip()]
    if not sep or not key or not raws:
        raise ValueError(f"--sweep must look like key.path=v1,v2,..., got {spec!r}")
    if len(set(raws)) != len(raws):
        raise ValueError(f"--sweep {key}: duplicate values in {raws}")
    return key, raws


def set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    """d["a"]["b"]["c"] = value for key "a.b.c", creating intermediate mappings."""
    parts = key.split(".")
    node = d
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"cannot set {key}: {part} is {child!r}, not a mapping")
        node = child
    node[parts[-1]] = value


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-")


@dataclass
class Variant:
    """One point of the sweep (index is 1-based: the SGE array task id)."""

    index: int
    name: str
    overrides: dict[str, Any]
    config: str
    run_dir: str
    trainable: bool = True
    # How the eval CLIs load this variant: --run <run_dir>, or --pos-mode ... for training-free modes.
    model_args: list[str] = field(default_factory=list)


def eval_model_args(pos: dict[str, Any], model: str) -> list[str]:
    """Eval CLI flags for a training-free `pos` block (none / pi / yarn).

    The flags are turned back into a pos dict exactly as the eval scripts do and must give the
    same settings; anything the flags cannot express (e.g. yarn.truncate) is an error, so a
    baseline is never silently evaluated with different settings than its config says.
    """
    from posext.eval.common import pos_from_args
    from posext.models.positional import PosConfig

    want = PosConfig.from_dict(pos)
    ns = argparse.Namespace(pos_mode=want.mode, factor=4.0, beta_fast=32.0, beta_slow=1.0, attention_factor=None)
    args = ["--model", model, "--pos-mode", want.mode]
    if want.mode == "pi":
        ns.factor = want.pi_factor
        args += ["--factor", repr(float(want.pi_factor))]
    elif want.mode == "yarn":
        y = want.yarn
        ns.factor, ns.beta_fast, ns.beta_slow, ns.attention_factor = y.factor, y.beta_fast, y.beta_slow, y.attention_factor
        args += ["--factor", repr(float(y.factor)), "--beta-fast", repr(float(y.beta_fast)),
                 "--beta-slow", repr(float(y.beta_slow))]
        if y.attention_factor is not None:
            args += ["--attention-factor", repr(float(y.attention_factor))]
    elif want.mode != "none":
        raise ValueError(f"pos.mode={want.mode!r} is trainable; eval it with --run")
    got = PosConfig.from_dict(pos_from_args(ns))
    if got.to_dict() != want.to_dict():
        raise ValueError(
            f"pos settings not expressible with eval CLI flags {args}:\n  config: {want.to_dict()}\n"
            f"  flags:  {got.to_dict()}"
        )
    return args


def expand_variants(
    base: dict[str, Any], axes: Sequence[tuple[str, list[str]]], sweep_dir: Path
) -> tuple[list[Variant], list[dict[str, Any]]]:
    """Cartesian product of the sweep axes; returns variants and their validated raw configs.

    Only axes with more than one value appear in variant names; single-value axes are plain
    overrides of the base config.
    """
    keys = [k for k, _ in axes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate --sweep keys: {keys}")
    for key in keys:
        if key in ("name", "output_dir"):
            raise ValueError(f"{key} is set by the pipeline; it cannot be swept")
    base_name = str(base.get("name") or "run")
    varying = [k for k, vals in axes if len(vals) > 1]
    short = {k: k.split(".")[-1] for k in varying}
    if len(set(short.values())) != len(short):  # e.g. seed and train.seed
        short = {k: k for k in varying}

    variants: list[Variant] = []
    raws: list[dict[str, Any]] = []
    for i, combo in enumerate(itertools.product(*(vals for _, vals in axes)), start=1):
        chosen = dict(zip(keys, combo))
        parts = [f"{_slug(short[k])}{_slug(chosen[k])}" for k in varying]
        name = f"{base_name}__{'_'.join(parts)}" if parts else base_name
        raw = copy.deepcopy(base)
        overrides = {k: parse_value(v) for k, v in chosen.items()}
        for k, v in overrides.items():
            set_dotted(raw, k, v)
        run_dir = sweep_dir / "runs" / name
        raw["name"] = name
        raw["output_dir"] = str(run_dir)
        if isinstance(raw.get("wandb"), dict):
            raw["wandb"]["run_name"] = name
        cfg = config_from_dict(raw)  # rejects unknown keys and bad pos settings
        trainable = cfg.pos.get("mode", "none") not in TRAINING_FREE_MODES
        if trainable:
            if cfg.train.steps < 1:
                raise ValueError(f"{name}: train.steps must be >= 1")
            model_args = ["--run", str(run_dir)]
        else:
            model_args = eval_model_args(cfg.pos, cfg.model)
        variants.append(
            Variant(i, name, overrides, str(sweep_dir / "configs" / f"{name}.yaml"), str(run_dir),
                    trainable, model_args)
        )
        raws.append(raw)
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"variant names collide: {names}")
    return variants, raws


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Plan dict (stored as plan.json) + raw config per variant."""
    base_path = Path(args.config)
    base = load_raw_config(base_path)
    base.setdefault("name", base_path.stem)
    name = _slug(args.name or str(base["name"]))
    sweep_dir = SWEEP_ROOT / name
    axes = [parse_sweep(s) for s in args.sweep]
    variants, raws = expand_variants(base, axes, sweep_dir)
    lengths = [int(x) for x in args.ppl_lengths.split(",") if x]
    plan = {
        "name": name,
        "sweep_dir": str(sweep_dir),
        "results_dir": str(RESULTS_ROOT / name),
        "base_config": str(base_path),
        "base": base,
        "axes": {k: v for k, v in axes},
        "variants": [asdict(v) for v in variants],
        "selection": {"metric": args.metric, "ppl_length": max(lengths)},
        "eval": {
            "ppl_lengths": lengths,
            "ppl_max_sequences": args.ppl_max_sequences,
            "passkey_n_per_cell": args.passkey_n_per_cell,
            "val_passkey_seed": VAL_PASSKEY_SEED,
            "test_passkey_seed": TEST_PASSKEY_SEED,
            "ruler": not args.no_ruler,
            "ruler_tasks": args.ruler_tasks,
            "ruler_lengths": args.ruler_lengths,
            "ruler_max_samples": args.ruler_max_samples,
        },
    }
    return plan, raws


def load_plan(sweep_dir: str | Path) -> dict[str, Any]:
    """Read outputs/sweeps/<name>/plan.json."""
    return json.loads((Path(sweep_dir) / "plan.json").read_text())


def _plan_core(plan: dict[str, Any]) -> dict[str, Any]:
    """The part of a plan that defines the experiment (not submission bookkeeping), JSON-normalized."""
    return json.loads(json.dumps({k: v for k, v in plan.items() if k not in ("jobs", "submitted")}))


def write_plan(plan: dict[str, Any], raws: list[dict[str, Any]]) -> None:
    """Write plan.json and the variant configs; refuses to redefine an existing sweep."""
    sweep_dir = Path(plan["sweep_dir"])
    plan_file = sweep_dir / "plan.json"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())
        if _plan_core(old) != _plan_core(plan):
            raise SystemExit(
                f"{plan_file} exists with a different plan. Use another --name, or delete "
                f"{sweep_dir} if that sweep is not needed."
            )
        logger.info("Sweep %s exists with the same plan: resubmitting (finished stages are skipped)", sweep_dir)
    for v, raw in zip(plan["variants"], raws):
        path = Path(v["config"])
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Generated by posext.pipeline (sweep {plan['name']}) from {plan['base_config']}\n"
        header += f"# overrides: {json.dumps(v['overrides'])}\n"
        path.write_text(header + yaml.safe_dump(raw, sort_keys=False))
    (sweep_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_json(plan_file, plan)


# ----------------------------------------------------------------------------------------
# Submission.


def hms(hours: float) -> str:
    """SGE h_rt string."""
    minutes = max(1, round(hours * 60))
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def ruler_test_ready(tasks: str | None, lengths: str | None) -> bool:
    """Whether every RULER test file has matching metadata (cheap; no data is read)."""
    from posext.eval.ruler import (
        DEFAULT_LENGTHS,
        DEFAULT_NUM_SAMPLES,
        DEFAULT_TASKS,
        SPLIT_SEEDS,
        data_file,
        default_data_root,
    )
    from posext.models.build import DEFAULT_MODEL

    root = default_data_root(DEFAULT_MODEL)
    task_list = tasks.split(",") if tasks else list(DEFAULT_TASKS)
    length_list = [int(x) for x in lengths.split(",")] if lengths else list(DEFAULT_LENGTHS)
    for length in length_list:
        for task in task_list:
            path = data_file(root, length, task, "test")
            meta_path = path.with_suffix(".meta.json")
            if not path.is_file() or not meta_path.is_file():
                return False
            meta = json.loads(meta_path.read_text())
            if (meta.get("seed"), meta.get("split"), meta.get("num_samples")) != (
                SPLIT_SEEDS["test"], "test", DEFAULT_NUM_SAMPLES["test"]
            ):
                return False
    return True


def _forward_env(extra: dict[str, str]) -> str:
    env = {k: os.environ[k] for k in FORWARD_ENV if os.environ.get(k)}
    env.setdefault("POSEXT_ROOT", os.getcwd())
    env.update(extra)
    for k, v in env.items():
        if "," in v:
            raise ValueError(f"{k}={v!r}: qsub -v values cannot contain commas")
    return ",".join(f"{k}={v}" for k, v in env.items())


def _job_name(sweep: str, stage: str) -> str:
    # SGE names: no leading digit, no / : @ \ * ?
    return f"sw-{stage}-{_slug(sweep)}"[:60]


def qsub_commands(plan: dict[str, Any], args: argparse.Namespace, *, prepare_ruler: bool) -> dict[str, list[str]]:
    """The qsub argv per stage, in submission order. `{train}` etc. are filled with job ids."""
    sweep, sweep_dir = plan["name"], plan["sweep_dir"]
    log_dir = f"{sweep_dir}/logs/"
    gpu = list(GPU_RESOURCES) + (["-l", f"gpu_type={args.gpu_type}"] if args.gpu_type else [])

    def base(stage: str, hours: float, resources: Sequence[str], hold: Sequence[str] = ()) -> list[str]:
        cmd = ["qsub", "-terse", "-cwd", "-j", "y", "-o", log_dir, "-N", _job_name(sweep, stage),
               "-l", f"h_rt={hms(hours)}", *resources]
        if hold:
            cmd += ["-hold_jid", ",".join(hold)]
        return cmd

    def stage_env(stage: str) -> list[str]:
        return ["-v", _forward_env({"SWEEP_DIR": sweep_dir, "STAGE": stage}), str(STAGE_SCRIPT)]

    n = len(plan["variants"])
    cmds: dict[str, list[str]] = {}
    train_hours = args.train_hours
    if train_hours is None:  # validation evals alone take minutes; shorter requests start sooner
        train_hours = 6.0 if any(v.get("trainable", True) for v in plan["variants"]) else 1.0
    train = base("train", train_hours, gpu) + ["-t", f"1-{n}"]
    if args.max_concurrent:
        train += ["-tc", str(args.max_concurrent)]
    cmds["train"] = train + stage_env("variant")
    if prepare_ruler:
        # qsub -v separates variables with commas, so lists go as a:b (ruler_prepare.qsub converts).
        env = {"SPLIT": "test"}
        if args.ruler_tasks:
            env["TASKS"] = args.ruler_tasks.replace(",", ":")
        if args.ruler_lengths:
            env["LENGTHS"] = args.ruler_lengths.replace(",", ":")
        # Resources and wall time come from the #$ lines of ruler_prepare.qsub.
        cmds["ruler_prepare"] = [
            "qsub", "-terse", "-cwd", "-j", "y", "-o", log_dir, "-N", _job_name(sweep, "rulerprep"),
            "-v", _forward_env(env), str(PREPARE_SCRIPT),
        ]
    cmds["select"] = base("select", 0.5, CPU_RESOURCES, ["{train}"]) + stage_env("select")
    cmds["test"] = base("test", args.test_hours, gpu, ["{select}"]) + stage_env("test")
    if plan["eval"]["ruler"]:
        hold = ["{select}"] + (["{ruler_prepare}"] if prepare_ruler else [])
        n_len = len(ruler_lengths(plan))
        cmds["ruler"] = base("ruler", args.ruler_hours, gpu, hold) + ["-t", f"1-{n_len}"] + stage_env("ruler")
        cmds["ruler_merge"] = base("rulermerge", 0.5, CPU_RESOURCES, ["{ruler}"]) + stage_env("ruler_merge")
    return cmds


def run_qsub(cmd: list[str]) -> str:
    """Submit; return the job id (array ids like 123.1-3:1 -> 123)."""
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    job_id = out.split(".")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected qsub output: {out!r}")
    return job_id


def submit(args: argparse.Namespace, *, qsub: Callable[[list[str]], str] = run_qsub) -> dict[str, Any]:
    """Build the plan, write it, and queue every stage with dependencies."""
    if not Path("posext/pipeline.py").is_file() or not STAGE_SCRIPT.is_file():
        raise SystemExit("run from the repo root (scripts/scc/pipeline.sh does this)")
    plan, raws = build_plan(args)
    prepare_ruler = False
    if plan["eval"]["ruler"]:
        from posext.eval.ruler import default_ruler_root

        if not args.dry_run and not (default_ruler_root() / "scripts").is_dir():
            raise SystemExit(
                f"No RULER checkout at {default_ruler_root()}: run scripts/scc/ruler_setup.sh on a "
                "login node first, or pass --no-ruler"
            )
        prepare_ruler = not ruler_test_ready(args.ruler_tasks, args.ruler_lengths)
    cmds = qsub_commands(plan, args, prepare_ruler=prepare_ruler)

    print(f"Sweep {plan['name']}: {len(plan['variants'])} variant(s) of {plan['base_config']}")
    for v in plan["variants"]:
        print(f"  [{v['index']}] {v['name']}  {json.dumps(v['overrides'])}")
    print(f"Selection: {plan['selection']['metric']} on validation; test evals on the winner only")
    if plan["eval"]["ruler"]:
        print("RULER test data: " + ("will be generated first (ruler_prepare job)" if prepare_ruler else "ready"))

    if args.dry_run:
        for stage, cmd in cmds.items():
            print(f"\n# {stage}\n" + " ".join(cmd))
        print("\n(dry run: nothing written or submitted)")
        return plan

    write_plan(plan, raws)
    jobs: dict[str, str] = {}
    try:
        for stage, cmd in cmds.items():
            filled = [part.format(**jobs) if "{" in part else part for part in cmd]
            jobs[stage] = qsub(filled)
            print(f"submitted {stage:14s} job {jobs[stage]}")
    finally:
        plan["jobs"] = jobs
        plan["submitted"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(Path(plan["sweep_dir"]) / "plan.json", plan)
        if len(jobs) < len(cmds):
            print(f"Submission stopped early; already queued: {jobs}. qdel them before retrying.")
    print(f"\nLogs:    {plan['sweep_dir']}/logs/\nStatus:  uv run python -m posext.pipeline status {plan['sweep_dir']}")
    return plan


# ----------------------------------------------------------------------------------------
# Stages (run inside jobs).


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _ppl_args(plan: dict[str, Any], split: str) -> list[str]:
    ev = plan["eval"]
    return ["--split", split, "--lengths", ",".join(map(str, ev["ppl_lengths"])),
            "--max-sequences", str(ev["ppl_max_sequences"]), "--device", "cuda"]


def model_args(v: dict[str, Any]) -> list[str]:
    """Eval CLI flags selecting a variant (plans written before baselines were supported: --run)."""
    return list(v.get("model_args") or ["--run", v["run_dir"]])


def _passkey_args(plan: dict[str, Any], seed: int) -> list[str]:
    ev = plan["eval"]
    return ["--seed", str(seed), "--n-per-cell", str(ev["passkey_n_per_cell"]), "--device", "cuda"]


def stage_variant(sweep_dir: Path, index: int, *, run: Callable[[list[str]], None] = _run) -> None:
    """Array task: train one variant, then validation PPL and passkey. Skips finished parts."""
    plan = load_plan(sweep_dir)
    v = plan["variants"][index - 1]
    assert v["index"] == index, (v, index)
    run_dir = Path(v["run_dir"])
    logger.info("Variant %d/%d: %s", index, len(plan["variants"]), v["name"])
    if not v.get("trainable", True):
        logger.info("Training-free (%s): no training", " ".join(model_args(v)))
    elif (run_dir / "train_summary.json").is_file():
        logger.info("Already trained: %s", run_dir)
    else:
        run(_py("posext.train", "--config", v["config"], "--device", "cuda", "--no-wandb", "--resume"))
    if v.get("trainable", True) and not (run_dir / "train_summary.json").is_file():
        raise RuntimeError(f"training did not finish: no {run_dir / 'train_summary.json'}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "val_ppl.json"
    if not out.is_file():
        run(_py("posext.eval.ppl", *model_args(v), *_ppl_args(plan, "validation"), "--out", str(out)))
    out = run_dir / "val_passkey.json"
    if not out.is_file():
        seed = plan["eval"]["val_passkey_seed"]
        run(_py("posext.eval.passkey", *model_args(v), *_passkey_args(plan, seed), "--out", str(out)))


def _zscores(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return [0.0 if std < 1e-8 else (x - mean) / std for x in values]


def rank_variants(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Add a `score` (higher is better) to each row and sort; ties go to lower NLL, then index.

    With two candidates that each win one metric, `combined` is always a tie (lower NLL wins).

    `combined` is the YaRN grid's score: 0.5 * z(-NLL) + 0.5 * z(passkey accuracy), z-scores
    over the candidates (posext.baselines.yarn_grid.combined_score).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    if metric == "combined":
        z_nll = _zscores([-r["nll"] for r in rows])
        z_acc = _zscores([r["passkey_accuracy"] for r in rows])
        scores = [0.5 * a + 0.5 * b for a, b in zip(z_nll, z_acc)]
    elif metric == "ppl":
        scores = [-r["nll"] for r in rows]
    else:
        scores = [r["passkey_accuracy"] for r in rows]
    # Round so exact ties (e.g. two candidates that each win one metric: z = +1 / -1 and -1 / +1)
    # reach the tie-breakers instead of being decided by float noise around 0.
    ranked = [{**r, "score": round(s, 9)} for r, s in zip(rows, scores)]
    ranked.sort(key=lambda r: (-r["score"], r["nll"], r["index"]))
    return ranked


def stage_select(sweep_dir: Path) -> dict[str, Any]:
    """Rank variants that finished validation; write selection.json."""
    plan = load_plan(sweep_dir)
    length = str(plan["selection"]["ppl_length"])
    rows, missing = [], []
    for v in plan["variants"]:
        run_dir = Path(v["run_dir"])
        ppl, pk = _read_json(run_dir / "val_ppl.json"), _read_json(run_dir / "val_passkey.json")
        if ppl is None or pk is None:
            missing.append(v["name"])
            continue
        rows.append({
            "index": v["index"], "name": v["name"], "run_dir": v["run_dir"], "overrides": v["overrides"],
            "model_args": model_args(v),
            "nll": ppl["lengths"][length]["nll"], "ppl": ppl["lengths"][length]["ppl"],
            "ppl_by_length": {k: s["ppl"] for k, s in ppl["lengths"].items()},
            "passkey_accuracy": pk["accuracy"], "passkey_by_length": pk["by_length"],
        })
    if missing:
        logger.warning("No validation results for %d variant(s): %s (see logs/)", len(missing), missing)
    if not rows:
        raise SystemExit(f"No variant of {plan['name']} finished validation; nothing to select")
    ranked = rank_variants(rows, plan["selection"]["metric"])
    selection = {
        "sweep": plan["name"],
        "metric": plan["selection"]["metric"],
        "ppl_length": int(length),
        "best": ranked[0],
        "ranking": ranked,
        "missing": missing,
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(Path(sweep_dir) / "selection.json", selection)
    logger.info("Selected %s (score %.3f, val PPL@%s %.3f, passkey %.3f)", ranked[0]["name"],
                ranked[0]["score"], length, ranked[0]["ppl"], ranked[0]["passkey_accuracy"])
    return selection


def _best(sweep_dir: Path) -> dict[str, Any]:
    sel = _read_json(Path(sweep_dir) / "selection.json")
    if sel is None:
        raise SystemExit(f"No {sweep_dir}/selection.json: the select stage did not finish")
    return sel["best"]


def test_paths(plan: dict[str, Any], variant: str) -> dict[str, Path]:
    """Where the test results of a variant are written."""
    root = Path(plan["results_dir"])
    return {k: root / f"test_{k}_{variant}.json" for k in ("ppl", "passkey", "ruler")}


def stage_test(sweep_dir: Path, *, run: Callable[[list[str]], None] = _run) -> None:
    """PG-19 test PPL and test passkey (seed 1) for the selected variant."""
    plan, best = load_plan(sweep_dir), _best(sweep_dir)
    paths = test_paths(plan, best["name"])
    if not paths["ppl"].is_file():
        run(_py("posext.eval.ppl", *model_args(best), *_ppl_args(plan, "test"), "--out", str(paths["ppl"])))
    if not paths["passkey"].is_file():
        seed = plan["eval"]["test_passkey_seed"]
        run(_py("posext.eval.passkey", *model_args(best), *_passkey_args(plan, seed), "--out", str(paths["passkey"])))


def retest_path(plan: dict[str, Any], variant: str, n_per_cell: int, prompt: str = "v1") -> Path:
    """Extra test passkey result next to the pipeline's own test results.

    v1 (original prompt) keeps its first name, test_passkey_n<N>_<variant>.json; other prompt
    versions get their own file, test_passkey_<prompt>_n<N>_<variant>.json.
    """
    tag = f"n{n_per_cell}" if prompt == "v1" else f"{prompt}_n{n_per_cell}"
    return Path(plan["results_dir"]) / f"test_passkey_{tag}_{variant}.json"


def stage_retest(sweep_dir: Path, n_per_cell: int, prompt: str = "v1", *,
                 run: Callable[[list[str]], None] = _run) -> None:
    """Test passkey (test seed) for the selected variant with `n_per_cell` prompts per cell."""
    plan, best = load_plan(sweep_dir), _best(sweep_dir)
    out = retest_path(plan, best["name"], n_per_cell, prompt)
    if out.is_file():
        logger.info("Already done: %s", out)
        return
    seed = plan["eval"]["test_passkey_seed"]
    run(_py("posext.eval.passkey", *model_args(best), "--seed", str(seed), "--n-per-cell", str(n_per_cell),
            "--prompt", prompt, "--device", "cuda", "--out", str(out)))


def job_queued(job_id: str) -> bool:
    """Whether an SGE job is still queued or running (qstat -j fails once it has finished)."""
    return subprocess.run(["qstat", "-j", str(job_id)], check=False, capture_output=True).returncode == 0


def retest(args: argparse.Namespace, *, qsub: Callable[[list[str]], str] = run_qsub,
           queued: Callable[[str], bool] = job_queued) -> dict[str, str]:
    """Queue one passkey re-test job per sweep (waits for the sweep's select job if still queued)."""
    if not STAGE_SCRIPT.is_file():
        raise SystemExit("run from the repo root (scripts/scc/pipeline.sh does this)")
    n, prompt = args.passkey_n_per_cell, args.passkey_prompt
    gpu = list(GPU_RESOURCES) + (["-l", f"gpu_type={args.gpu_type}"] if args.gpu_type else [])
    submitted: dict[str, str] = {}
    for sweep_dir in args.sweep_dirs:
        plan = load_plan(sweep_dir)
        sel = _read_json(Path(sweep_dir) / "selection.json")
        select_job = (plan.get("jobs") or {}).get("select")
        hold = [select_job] if select_job and queued(select_job) else []
        if sel is None and not hold:
            print(f"skip {plan['name']}: no selection.json and its select job is not queued (failed?)")
            continue
        if sel is not None and not hold and retest_path(plan, sel["best"]["name"], n, prompt).is_file():
            print(f"skip {plan['name']}: already has {retest_path(plan, sel['best']['name'], n, prompt)}")
            continue
        cmd = ["qsub", "-terse", "-cwd", "-j", "y", "-o", f"{plan['sweep_dir']}/logs/",
               "-N", _job_name(plan["name"], f"retest{n}" if prompt == "v1" else f"retest{prompt}n{n}"),
               "-l", f"h_rt={hms(args.hours)}", *gpu]
        if hold:
            cmd += ["-hold_jid", ",".join(hold)]
        cmd += ["-v", _forward_env({"SWEEP_DIR": plan["sweep_dir"], "STAGE": "retest", "PASSKEY_N": str(n),
                                    "PASSKEY_PROMPT": prompt}),
                str(STAGE_SCRIPT)]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        submitted[plan["name"]] = qsub(cmd)
        wait = f" (waits for select job {hold[0]})" if hold else ""
        print(f"submitted retest {plan['name']:32s} job {submitted[plan['name']]}{wait}")
    if args.dry_run:
        print("(dry run: nothing submitted)")
    return submitted


def ruler_lengths(plan: dict[str, Any]) -> list[int]:
    """RULER lengths of a plan (default: RULER's 4k / 8k / 16k)."""
    from posext.eval.ruler import DEFAULT_LENGTHS

    given = plan["eval"].get("ruler_lengths")
    return [int(x) for x in given.split(",") if x] if given else list(DEFAULT_LENGTHS)


def ruler_tasks(plan: dict[str, Any]) -> list[str]:
    """RULER tasks of a plan (default: the 8 NIAH tasks)."""
    from posext.eval.ruler import DEFAULT_TASKS

    given = plan["eval"].get("ruler_tasks")
    return [x for x in given.split(",") if x] if given else list(DEFAULT_TASKS)


def _ruler_pred_dir(sweep_dir: Path, variant: str) -> Path:
    return Path(sweep_dir) / "ruler_test" / variant


def stage_ruler(sweep_dir: Path, index: int, *, run: Callable[[list[str]], None] = _run) -> None:
    """Array task: RULER test split at one length (index is 1-based) for the selected variant.

    Lengths run as separate jobs so 16k does not hold up 4k / 8k; all write into one prediction
    dir (one file per task and length). Resumable.
    """
    plan, best = load_plan(sweep_dir), _best(sweep_dir)
    length = ruler_lengths(plan)[index - 1]
    pred_dir = _ruler_pred_dir(sweep_dir, best["name"])
    out = pred_dir / f"result_{length}.json"
    done = _read_json(out)
    if done is not None and done.get("complete"):
        logger.info("RULER @ %d already complete: %s", length, out)
        return
    cmd = _py("posext.eval.ruler", "run", *model_args(best), "--split", "test", "--device", "cuda",
              "--tag", best["name"], "--pred-dir", str(pred_dir), "--out", str(out),
              "--tasks", ",".join(ruler_tasks(plan)), "--lengths", str(length))
    if plan["eval"].get("ruler_max_samples"):
        cmd += ["--max-samples", str(plan["eval"]["ruler_max_samples"])]
    run(cmd)


def stage_ruler_merge(sweep_dir: Path) -> dict[str, Any]:
    """Score all lengths' predictions together into the final RULER test result."""
    from posext.eval.ruler import score_ruler

    plan, best = load_plan(sweep_dir), _best(sweep_dir)
    pred_dir = _ruler_pred_dir(sweep_dir, best["name"])
    results = score_ruler(pred_dir, tasks=ruler_tasks(plan), lengths=ruler_lengths(plan))
    results.update({"tag": best["name"], "source": model_args(best), "pred_dir": str(pred_dir),
                    "max_samples": plan["eval"].get("ruler_max_samples")})
    out = test_paths(plan, best["name"])["ruler"]
    _write_json(out, results)
    if not results["complete"]:
        logger.warning("RULER incomplete for %s (a length job failed or hit wall time?): %s", best["name"], out)
    else:
        logger.info("RULER avg %.2f %s -> %s", results["avg"], results["avg_by_length"], out)
    return results


def status(sweep_dir: Path) -> str:
    """Human-readable progress of a sweep."""
    plan = load_plan(sweep_dir)
    length = str(plan["selection"]["ppl_length"])
    sel = _read_json(Path(sweep_dir) / "selection.json")
    best = sel["best"]["name"] if sel else None
    lines = [f"Sweep {plan['name']}  (submitted {plan.get('submitted', '-')}, jobs {plan.get('jobs', {})})",
             f"{'variant':60s} {'trained':>7s} {'valPPL@' + length:>11s} {'passkey':>8s}"]
    for v in plan["variants"]:
        run_dir = Path(v["run_dir"])
        trained = "yes" if (run_dir / "train_summary.json").is_file() else (
            "partial" if (run_dir / "train_state.pt").is_file() else "-")
        ppl, pk = _read_json(run_dir / "val_ppl.json"), _read_json(run_dir / "val_passkey.json")
        ppl_s = f"{ppl['lengths'][length]['ppl']:.3f}" if ppl else "-"
        pk_s = f"{pk['accuracy']:.3f}" if pk else "-"
        mark = "  <- selected" if v["name"] == best else ""
        lines.append(f"{v['name']:60s} {trained:>7s} {ppl_s:>11s} {pk_s:>8s}{mark}")
    if best:
        paths = test_paths(plan, best)
        lines.append(f"\nTest results for {best}:")
        for key, path in paths.items():
            res = _read_json(path)
            try:
                if res is None and key == "ruler" and not plan["eval"].get("ruler", True):
                    summary = "not run (--no-ruler)"
                elif res is None and key == "ruler":
                    lens = [p.stem.split("_")[-1] for p in sorted(_ruler_pred_dir(sweep_dir, best).glob("result_*.json"))]
                    summary = "pending" + (f" (lengths with results: {', '.join(lens)})" if lens else "")
                elif res is None:
                    summary = "pending"
                elif key == "ppl":
                    summary = "  ".join(f"PPL@{k}={s['ppl']:.3f}" for k, s in res["lengths"].items())
                elif key == "passkey":
                    summary = f"acc={res['accuracy']:.3f}  by length {res['by_length']}"
                else:
                    summary = f"avg={res['avg']}  by length {res['avg_by_length']}  complete={res['complete']}"
            except (KeyError, TypeError, ValueError):
                summary = "written (unexpected format)"
            lines.append(f"  {key:8s} {summary}   ({path})")
        extra = sorted(Path(plan["results_dir"]).glob(f"test_passkey_*_{best}.json"))
        for path in [x for x in extra if x != paths["passkey"]]:
            res = _read_json(path)
            n = path.name[len("test_passkey_"):-len(f"_{best}.json")]
            try:
                lines.append(f"  passkey {n}: acc={res['accuracy']:.3f}  (n={res['n']})  by length {res['by_length']}")
            except (KeyError, TypeError):
                lines.append(f"  passkey {n}: written (unexpected format)   ({path})")
    else:
        lines.append("\nNot selected yet.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# CLI.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI: submit (login node), variant / select / test / ruler (inside jobs), status."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit", help="write the plan and queue every stage")
    sp.add_argument("config", type=Path, help="base experiment YAML (trainable mode, or none/pi/yarn baseline)")
    sp.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2",
                    help="dotted config key and values; repeat for a grid (e.g. train.lr=1e-3,3e-2)")
    sp.add_argument("--name", default=None, help="sweep name (default: the config's name)")
    sp.add_argument("--metric", default="combined", choices=METRICS, help="validation selection score")
    sp.add_argument("--ppl-lengths", default="4096,8192,16384")
    sp.add_argument("--ppl-max-sequences", type=int, default=50)
    sp.add_argument("--passkey-n-per-cell", type=int, default=10, help="prompts per (length, depth)")
    sp.add_argument("--no-ruler", action="store_true", help="skip RULER test evaluation")
    sp.add_argument("--ruler-tasks", default=None, help="default: the 8 NIAH tasks")
    sp.add_argument("--ruler-lengths", default=None, help="default: 4096,8192,16384")
    sp.add_argument("--ruler-max-samples", type=int, default=None,
                    help="first N test samples per task and length (smoke runs only, not for results)")
    sp.add_argument("--gpu-type", default=None, help="pin all GPU jobs to one type, e.g. A100 or L40S")
    sp.add_argument("--max-concurrent", type=int, default=None, help="max training tasks running at once")
    sp.add_argument("--train-hours", type=float, default=None,
                    help="wall time per variant (train + val); default 6, or 1 if nothing is trained")
    sp.add_argument("--test-hours", type=float, default=4.0, help="wall time for test PPL + passkey")
    sp.add_argument("--ruler-hours", type=float, default=12.0, help="wall time per RULER length job")
    sp.add_argument("--dry-run", action="store_true", help="print plan and qsub commands; submit nothing")

    sp = sub.add_parser("retest", help="queue test passkey with more prompts for each sweep's selected model")
    sp.add_argument("sweep_dirs", nargs="+", type=Path, help="outputs/sweeps/<name> (globs are fine)")
    sp.add_argument("--passkey-n-per-cell", type=int, required=True, help="prompts per (length, depth)")
    sp.add_argument("--passkey-prompt", default="v2", choices=("v1", "v2"),
                    help="v2 = standard prompt (default); v1 = original prompt")
    sp.add_argument("--hours", type=float, default=2.0, help="wall time per job")
    sp.add_argument("--gpu-type", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp = sub.add_parser("retest_stage", help="(job) test passkey with more prompts for one sweep")
    sp.add_argument("sweep_dir", type=Path)
    sp.add_argument("--n", type=int, required=True)
    sp.add_argument("--prompt", default="v1", choices=("v1", "v2"))

    sp = sub.add_parser("variant", help="(job) train + validate one variant")
    sp.add_argument("sweep_dir", type=Path)
    sp.add_argument("--index", type=int, required=True, help="1-based ($SGE_TASK_ID)")
    sp = sub.add_parser("ruler", help="(job) RULER test at one length")
    sp.add_argument("sweep_dir", type=Path)
    sp.add_argument("--index", type=int, required=True, help="1-based length index ($SGE_TASK_ID)")
    for name in ("select", "test", "ruler_merge", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("sweep_dir", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.cmd == "submit":
        submit(args)
    elif args.cmd == "retest":
        retest(args)
    elif args.cmd == "retest_stage":
        stage_retest(args.sweep_dir, args.n, args.prompt)
    elif args.cmd == "variant":
        stage_variant(args.sweep_dir, args.index)
    elif args.cmd == "select":
        stage_select(args.sweep_dir)
    elif args.cmd == "test":
        stage_test(args.sweep_dir)
    elif args.cmd == "ruler":
        stage_ruler(args.sweep_dir, args.index)
    elif args.cmd == "ruler_merge":
        stage_ruler_merge(args.sweep_dir)
    else:
        print(status(args.sweep_dir))


if __name__ == "__main__":
    main()
