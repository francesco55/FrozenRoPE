"""Training curves (loss, gradient norm) + summary table from training job logs.

    # sweeps made by scripts/scc/pipeline.sh (one panel per sweep, one line per setting)
    uv run --with matplotlib python -m posext.plot_training outputs/sweeps/temp_*
    uv run --with matplotlib python -m posext.plot_training outputs/sweeps/temp_head_* --name head
    # standalone training jobs (scripts/scc/train.qsub logs)
    uv run --with matplotlib python -m posext.plot_training logs/train_8576388.log logs/train_8576389.log

Writes to outputs/plots/<name>/ (gitignored; change with --out):
  loss.png       training loss vs step, one panel per sweep (shared y), selected setting in bold
  grad_norm.png  gradient norm (before clipping) vs step, log scale
  compare.png    each sweep's selected setting on one axis (when there is more than one sweep)
  curves.csv     every logged point: group, run, step, loss, grad_norm, seconds
  summary.md     per run: final loss, validation PPL / passkey, selection, GPU, minutes

Data comes from the `step k/N  loss=...  grad_norm=...` lines that posext.train logs every
`train.log_every` steps (W&B is off in SCC jobs). The logged loss is the mean over the steps
since the previous log line; grad_norm is the value at that step. If a run was resubmitted
(resumed), points from newer logs replace older ones at the same step. matplotlib is pulled in by
`uv run --with`, so the project's dependencies stay unchanged; parsing alone needs no matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Validated categorical palette in fixed order (same as the course HW plots); text/grid stay neutral.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"

TRAINING_RE = re.compile(r"Training (\S+) for (\d+) steps on \S+ \(regime=(\w+), seq_len=(\d+), target_len=(\d+)\)")
STEP_RE = re.compile(r"step (\d+)/(\d+)\s+loss=([-\d.naif]+)\s+\S+\s+grad_norm=([-\d.naif]+)\s+([\d.]+)s")


@dataclass
class Run:
    """One training run: logged points plus whatever the sweep recorded about it."""

    group: str  # sweep name, or "standalone"
    name: str  # full run name (config name)
    label: str  # short label within the group (e.g. lr1e-2)
    total_steps: int = 0
    regime: str = ""
    seq_len: int = 0
    points: dict[int, dict[str, float]] = field(default_factory=dict)  # step -> loss, grad_norm, seconds
    selected: bool = False
    val_ppl: dict[str, float] = field(default_factory=dict)  # length -> ppl
    val_passkey: float | None = None
    gpu: str | None = None
    minutes: float | None = None

    @property
    def steps(self) -> list[int]:
        return sorted(self.points)

    @property
    def complete(self) -> bool:
        return bool(self.points) and max(self.points) >= self.total_steps

    def final_loss(self, last: int = 5) -> float:
        """Mean logged loss over the last `last` points (each already a mean over log_every steps)."""
        vals = [self.points[s]["loss"] for s in self.steps[-last:] if math.isfinite(self.points[s]["loss"])]
        return statistics.mean(vals) if vals else math.nan


def parse_log(text: str) -> list[dict[str, Any]]:
    """Training segments in a log: name, total steps, regime, and points (a log may hold one run)."""
    runs: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = TRAINING_RE.search(line)
        if m:
            runs.append({"name": m[1], "total": int(m[2]), "regime": m[3], "seq_len": int(m[4]), "points": {}})
            continue
        m = STEP_RE.search(line)
        if m and runs:
            runs[-1]["points"][int(m[1])] = {
                "loss": float(m[3]), "grad_norm": float(m[4]), "seconds": float(m[5])}
    return runs


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _merge(run: Run, segment: dict[str, Any]) -> None:
    run.total_steps = segment["total"] or run.total_steps
    run.regime, run.seq_len = segment["regime"], segment["seq_len"]
    run.points.update(segment["points"])  # newer log wins at the same step


def _logs_oldest_first(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))


def load_sweep(sweep_dir: Path) -> list[Run]:
    """Runs of a pipeline sweep (outputs/sweeps/<name>): logs + plan + selection + val results."""
    plan = _read_json(sweep_dir / "plan.json")
    if plan is None:
        raise FileNotFoundError(f"{sweep_dir} has no plan.json (not a pipeline sweep?)")
    group = plan["name"]
    selection = _read_json(sweep_dir / "selection.json") or {}
    best = (selection.get("best") or {}).get("name")
    runs: dict[str, Run] = {}
    for v in plan["variants"]:
        if not v.get("trainable", True):
            continue  # baselines have nothing to plot
        label = v["name"].split("__", 1)[1] if "__" in v["name"] else v["name"]
        run = Run(group=group, name=v["name"], label=label, selected=v["name"] == best)
        run_dir = sweep_dir / "runs" / v["name"]  # = plan's run_dir; works on a copied sweep dir too
        ppl = _read_json(run_dir / "val_ppl.json")
        if ppl:
            run.val_ppl = {k: s["ppl"] for k, s in ppl["lengths"].items()}
        pk = _read_json(run_dir / "val_passkey.json")
        if pk:
            run.val_passkey = pk["accuracy"]
        summary = _read_json(run_dir / "train_summary.json")
        if summary:
            run.gpu = (summary.get("gpu_name") or "").replace("NVIDIA ", "") or None
            run.minutes = summary["seconds"] / 60
        runs[v["name"]] = run
    for log in _logs_oldest_first(list((sweep_dir / "logs").glob("sw-train-*"))):
        for segment in parse_log(log.read_text(errors="replace")):
            if segment["name"] in runs:
                _merge(runs[segment["name"]], segment)
    return list(runs.values())


def load_logs(paths: list[Path]) -> list[Run]:
    """Runs from standalone training logs (e.g. scripts/scc/train.qsub), grouped as 'standalone'."""
    runs: dict[str, Run] = {}
    for log in _logs_oldest_first(paths):
        for segment in parse_log(log.read_text(errors="replace")):
            run = runs.setdefault(segment["name"], Run(group="standalone", name=segment["name"], label=segment["name"]))
            _merge(run, segment)
    return list(runs.values())


def load(inputs: list[Path]) -> list[Run]:
    """Sweep dirs and/or log files -> runs that have at least one logged point."""
    runs: list[Run] = []
    logs = [p for p in inputs if p.is_file()]
    for p in inputs:
        if p.is_dir():
            runs += load_sweep(p)
        elif not p.is_file():
            raise FileNotFoundError(p)
    runs += load_logs(logs)
    return [r for r in runs if r.points]


def smooth(values: list[float], window: int) -> list[float]:
    """Trailing moving average (window 1 = unchanged)."""
    if window <= 1:
        return values
    return [statistics.mean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]


def write_csv(runs: list[Run], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "run", "label", "step", "loss", "grad_norm", "seconds"])
        for r in runs:
            for s in r.steps:
                p = r.points[s]
                w.writerow([r.group, r.name, r.label, s, p["loss"], p["grad_norm"], p["seconds"]])


def summary_table(runs: list[Run]) -> str:
    lengths = sorted({k for r in runs for k in r.val_ppl}, key=int)
    head = ["group", "run", "regime", "steps", "final train loss"] + [f"val PPL {int(k) // 1024}k" for k in lengths]
    head += ["val passkey", "selected", "GPU", "min"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        logged = f"{max(r.points)}/{r.total_steps}" + ("" if r.complete else " (stopped)")
        cells = [r.group, r.label, f"{r.regime} {r.seq_len // 1024}k" if r.regime else "", logged,
                 f"{r.final_loss():.4f}"]
        cells += [f"{r.val_ppl[k]:.2f}" if k in r.val_ppl else "" for k in lengths]
        cells += [f"{r.val_passkey:.3f}" if r.val_passkey is not None else "", "yes" if r.selected else "",
                  r.gpu or "", f"{r.minutes:.0f}" if r.minutes is not None else ""]
        rows.append("| " + " | ".join(cells) + " |")
    note = ("\nfinal train loss = mean of the last 5 logged points (each a mean over `train.log_every` "
            "steps); val = PG-19 validation PPL and passkey (seed 0) from the sweep.\n")
    return "\n".join(rows) + "\n" + note


# ----------------------------------------------------------------------------------------
# Plotting (matplotlib imported lazily so parsing and tests work without it).


def _plt():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("matplotlib is not installed: run with `uv run --with matplotlib python -m "
                         "posext.plot_training ...`") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _groups(runs: list[Run]) -> list[tuple[str, list[Run]]]:
    order: dict[str, list[Run]] = {}
    for r in runs:
        order.setdefault(r.group, []).append(r)
    return [(g, sorted(rs, key=lambda r: r.label)) for g, rs in order.items()]


def _label_colors(runs: list[Run]) -> dict[str, str]:
    """Same label -> same color in every panel (e.g. lr1e-2 is always blue)."""
    labels = sorted({r.label for r in runs})
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def _draw(ax, run: Run, metric: str, color: str, window: int, bold: bool, label: str) -> None:
    xs = run.steps
    ys = smooth([run.points[s][metric] for s in xs], window)
    ax.plot(xs, ys, color=color, lw=2.2 if bold else 1.2, alpha=1.0 if bold else 0.75, label=label)
    if not run.complete:  # stopped before its last step (failed / hit wall time)
        ax.plot(xs[-1], ys[-1], marker="x", ms=8, mew=2, color=color)


def plot_facets(runs: list[Run], metric: str, title: str, window: int, out: Path) -> None:
    """One panel per group (sweep), shared axes; one line per run, selected run in bold."""
    plt = _plt()
    groups = _groups(runs)
    colors = _label_colors(runs)
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (group, rs) in zip(axes.flat, groups):
        for r in rs:
            _draw(ax, r, metric, colors[r.label], window, r.selected, r.label)
        _style(ax, group)
        if metric == "grad_norm":
            ax.set_yscale("log")
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("training step", color=MUTED, fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in colors.values()]
    labels = list(colors)
    if any(r.selected for r in runs):
        handles.append(plt.Line2D([], [], color=INK, lw=2.2))
        labels.append("bold = selected on validation")
    if any(not r.complete for r in runs):
        handles.append(plt.Line2D([], [], marker="x", ls="", color=MUTED, mew=2, ms=8))
        labels.append("run stopped early")
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=8,
               labelcolor=INK)
    note = f" ({window}-point moving average)" if window > 1 else ""
    fig.suptitle(title + note, color=INK, fontsize=12, x=0.01, ha="left")
    rows_of_legend = math.ceil(len(handles) / 5)
    fig.tight_layout(rect=(0, 0.03 + 0.03 * rows_of_legend, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def plot_compare(runs: list[Run], window: int, out: Path) -> None:
    """Each group's selected run (or its only run) on one axis: loss | grad norm."""
    plt = _plt()
    picks = []
    for group, rs in _groups(runs):
        chosen = [r for r in rs if r.selected] or (rs if len(rs) == 1 else [])
        picks += [(f"{group} ({r.label})", r) for r in chosen]
    if len(picks) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (label, r) in enumerate(picks):
        color = PALETTE[i % len(PALETTE)]
        _draw(axes[0], r, "loss", color, window, True, label)
        _draw(axes[1], r, "grad_norm", color, window, True, label)
    _style(axes[0], "Training loss")
    _style(axes[1], "Gradient norm (before clipping)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("training step", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 3), frameon=False, fontsize=8,
               labelcolor=INK)
    note = f" ({window}-point moving average)" if window > 1 else ""
    fig.suptitle("Selected setting of each sweep" + note, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04 + 0.04 * math.ceil(len(handles) / 3), 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def default_name(inputs: list[Path]) -> str:
    """Output folder name: the common prefix of the inputs' names, else 'training'."""
    names = [p.name if p.is_dir() else p.stem for p in inputs]
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
    return prefix.rstrip("_-.") or "training"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="sweep dirs (outputs/sweeps/<name>) and/or training logs")
    p.add_argument("--name", default=None, help="output folder name (default: common prefix of the inputs)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: outputs/plots/<name>)")
    p.add_argument("--smooth", type=int, default=1, help="extra moving average over logged points (1 = off)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = load(args.inputs)
    if not runs:
        raise SystemExit("no training points found in the inputs (no `step k/N loss=...` lines)")
    out = args.out or Path("outputs/plots") / (args.name or default_name(args.inputs))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(runs, out / "curves.csv")
    table = summary_table(runs)
    (out / "summary.md").write_text(table)
    print(table)
    plot_facets(runs, "loss", "Training loss", args.smooth, out / "loss.png")
    plot_facets(runs, "grad_norm", "Gradient norm (before clipping, log scale)", args.smooth, out / "grad_norm.png")
    plot_compare(runs, args.smooth, out / "compare.png")


if __name__ == "__main__":
    main()
