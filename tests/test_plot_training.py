"""Tests for posext.plot_training: log parsing, sweep loading, summary, plots (if matplotlib)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from posext import plot_training as pt

STAMP = "2026-11-12 01:48:49,645 INFO posext.train: "


def _log(name: str, steps: list[int], total: int = 500, loss0: float = 3.0, regime: str = "long",
         seq_len: int = 16384, resumed: bool = False) -> str:
    """A training log in the exact format posext.train writes (plus SGE / pipeline noise)."""
    lines = ["Host=scc-506  JOB_ID=1  TASK=1  STAGE=variant  SWEEP_DIR=outputs/sweeps/x",
             f"{STAMP.replace('posext.train', 'posext.pipeline')}Variant 1/2: {name}"]
    if resumed:
        lines.append(f"{STAMP}Resumed from step {steps[0] - 10}")
    lines.append(f"{STAMP}Training {name} for {total} steps on cuda (regime={regime}, seq_len={seq_len}, "
                 f"target_len=16384)")
    for s in steps:
        loss = loss0 - 0.001 * s
        lines.append(f"{STAMP}step {s}/{total}  loss={loss:.4f}  ppl?{2.718 ** loss:.2f}  grad_norm={20 + s / 100:.3f}  "
                     f"{s * 1.9:.1f}s")
    lines.append(f"{STAMP}Saved checkpoint at step {steps[-1]} -> outputs/x")
    return "\n".join(lines) + "\n"


def test_parse_log() -> None:
    (seg,) = pt.parse_log(_log("rq1_x__lr1e-2", [1, 10, 20], total=20, regime="pose", seq_len=4096))
    assert (seg["name"], seg["total"], seg["regime"], seg["seq_len"]) == ("rq1_x__lr1e-2", 20, "pose", 4096)
    assert sorted(seg["points"]) == [1, 10, 20]
    assert seg["points"][10] == {"loss": pytest.approx(2.99), "grad_norm": pytest.approx(20.1),
                                 "seconds": pytest.approx(19.0)}
    assert pt.parse_log("no training here\n") == []


def test_parse_log_nan_loss() -> None:
    text = _log("r", [1]) + f"{STAMP}step 10/500  loss=nan  ppl?nan  grad_norm=inf  3.0s\n"
    (seg,) = pt.parse_log(text)
    assert seg["points"][10]["grad_norm"] == float("inf")


def _sweep(root: Path, name: str = "temp_head_init0", trainable: bool = True) -> Path:
    sd = root / "outputs" / "sweeps" / name
    variants = []
    for i, lr in enumerate(["1e-2", "3e-2"], start=1):
        vname = f"rq1_learned_freq_temp_head__lr{lr}"
        rd = sd / "runs" / vname
        rd.mkdir(parents=True)
        (rd / "val_ppl.json").write_text(json.dumps({"lengths": {"4096": {"ppl": 17.0 + i}, "16384": {"ppl": 15.0 + i}}}))
        (rd / "val_passkey.json").write_text(json.dumps({"accuracy": 0.5 + i / 10}))
        (rd / "train_summary.json").write_text(json.dumps({"seconds": 1080.0, "gpu_name": "NVIDIA L40S"}))
        variants.append({"index": i, "name": vname, "run_dir": f"outputs/sweeps/{name}/runs/{vname}",
                         "trainable": trainable})
    (sd / "plan.json").write_text(json.dumps({"name": name, "variants": variants}))
    (sd / "selection.json").write_text(json.dumps({"best": {"name": variants[1]["name"]}}))
    (sd / "logs").mkdir()
    return sd


def test_load_sweep_merges_resubmitted_logs(tmp_path: Path) -> None:
    sd = _sweep(tmp_path)
    a, b = "rq1_learned_freq_temp_head__lr1e-2", "rq1_learned_freq_temp_head__lr3e-2"
    old = sd / "logs" / "sw-train-temp_head_init0.o100.1"
    old.write_text(_log(a, [1, 10, 20, 30], loss0=9.0))  # first attempt, hit wall time at step 30
    new = sd / "logs" / "sw-train-temp_head_init0.o200.1"
    new.write_text(_log(a, [30, 40, 500], loss0=3.0, resumed=True))  # resubmission resumes
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    (sd / "logs" / "sw-train-temp_head_init0.o100.2").write_text(_log(b, [1, 10, 500]))
    (sd / "logs" / "sw-select-temp_head_init0.o101").write_text("Selected ...\n")  # not a train log

    runs = {r.name: r for r in pt.load([sd])}
    ra, rb = runs[a], runs[b]
    assert ra.steps == [1, 10, 20, 30, 40, 500] and ra.complete
    assert ra.points[30]["loss"] == pytest.approx(3.0 - 0.03)  # newer log wins at the shared step
    assert ra.points[10]["loss"] == pytest.approx(9.0 - 0.01)  # older-only steps are kept
    assert (ra.label, ra.group, ra.selected, rb.selected) == ("lr1e-2", "temp_head_init0", False, True)
    assert ra.val_ppl == {"4096": 18.0, "16384": 16.0} and ra.val_passkey == pytest.approx(0.6)
    assert (ra.gpu, ra.minutes) == ("L40S", 18.0)


def test_incomplete_run_and_baseline_sweeps(tmp_path: Path) -> None:
    sd = _sweep(tmp_path)
    (sd / "logs" / "sw-train-temp_head_init0.o1.1").write_text(_log("rq1_learned_freq_temp_head__lr1e-2", [1, 10]))
    (run,) = pt.load([sd])  # the other setting has no log yet: dropped
    assert not run.complete and "10/500 (stopped)" in pt.summary_table([run])
    base = _sweep(tmp_path, "rq1_yarn_tuned", trainable=False)
    assert pt.load([base]) == []  # baselines: nothing trained, nothing to plot


def test_standalone_logs_and_names(tmp_path: Path) -> None:
    a = tmp_path / "train_8576388.log"
    a.write_text(_log("rq1f_learned_freq_head_lr3e-2", [1, 10, 500]))
    b = tmp_path / "train_8576389.log"
    b.write_text(_log("rq1f_learned_freq_head_lr1e-1", [1, 10, 500]))
    runs = pt.load([a, b])
    assert {r.group for r in runs} == {"standalone"} and len(runs) == 2
    assert pt.default_name([a, b]) == "train_857638"
    assert pt.default_name([Path("outputs/sweeps/temp_head_init0"), Path("outputs/sweeps/temp_head_inityarn")]) == "temp_head_init"
    assert pt.default_name([Path("a"), Path("b")]) == "training"
    with pytest.raises(FileNotFoundError):
        pt.load([tmp_path / "missing.log"])


def test_summary_and_csv(tmp_path: Path) -> None:
    sd = _sweep(tmp_path)
    for i, lr in enumerate(["1e-2", "3e-2"], start=1):
        (sd / "logs" / f"sw-train-temp_head_init0.o1.{i}").write_text(
            _log(f"rq1_learned_freq_temp_head__lr{lr}", [1, 10, 20, 30, 40, 500]))
    runs = pt.load([sd])
    table = pt.summary_table(runs)
    assert "| temp_head_init0 | lr3e-2 | long 16k | 500/500 |" in table
    assert "val PPL 4k | val PPL 16k" in table and "| yes | L40S | 18 |" in table
    pt.write_csv(runs, tmp_path / "curves.csv")
    lines = (tmp_path / "curves.csv").read_text().splitlines()
    assert lines[0] == "group,run,label,step,loss,grad_norm,seconds" and len(lines) == 1 + 12
    assert pt.smooth([1.0, 2.0, 3.0, 4.0], 2) == [1.0, 1.5, 2.5, 3.5]


def test_plots_written(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    sweeps = []
    for name in ("temp_head_init0", "temp_head_inityarn"):
        sd = _sweep(tmp_path, name)
        for i, lr in enumerate(["1e-2", "3e-2"], start=1):
            (sd / "logs" / f"sw-train-{name}.o1.{i}").write_text(
                _log(f"rq1_learned_freq_temp_head__lr{lr}", [1, 10, 20, 500 if i == 2 else 30]))
        sweeps.append(sd)
    out = tmp_path / "plots"
    pt.main([*map(str, sweeps), "--out", str(out), "--smooth", "2"])
    for f in ("loss.png", "grad_norm.png", "compare.png", "curves.csv", "summary.md"):
        assert (out / f).stat().st_size > 0, f
