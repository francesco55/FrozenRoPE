"""CPU tests for the SCC sweep pipeline (posext.pipeline): plan, selection, job chaining, stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from posext import pipeline as pl
from posext.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "rq1" / "learned_freq_temp_head.yaml"


def _submit_args(*argv: str):
    return pl.parse_args(["submit", str(BASE), *argv])


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake repo root (cwd) with the files submit checks for; RULER root present, no test data."""
    (tmp_path / "posext").mkdir()
    (tmp_path / "posext" / "pipeline.py").write_text("")
    (tmp_path / "scripts" / "scc").mkdir(parents=True)
    (tmp_path / "scripts" / "scc" / "sweep_stage.qsub").write_text("")
    (tmp_path / "RULER" / "scripts").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSEXT_RULER_ROOT", str(tmp_path / "RULER"))
    monkeypatch.setenv("POSEXT_DATA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("POSEXT_ROOT", str(tmp_path))
    return tmp_path


def test_parse_value_and_sweep() -> None:
    assert pl.parse_value("1e-3") == pytest.approx(1e-3)  # YAML 1.1 would keep this a string
    assert pl.parse_value("500") == 500 and isinstance(pl.parse_value("500"), int)
    assert pl.parse_value("true") is True
    assert pl.parse_value("adascale") == "adascale"
    assert pl.parse_sweep("train.lr=1e-3, 3e-2") == ("train.lr", ["1e-3", "3e-2"])
    for bad in ("train.lr", "train.lr=", "=1,2", "train.lr=1,1"):
        with pytest.raises(ValueError):
            pl.parse_sweep(bad)


def test_expand_grid_names_and_configs(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE.read_text())
    axes = [("train.lr", ["1e-2", "3e-2"]), ("pos.temp.form", ["adascale", "constant"]), ("train.steps", ["200"])]
    variants, raws = pl.expand_variants(base, axes, tmp_path / "sw")
    assert [v.index for v in variants] == [1, 2, 3, 4]
    assert variants[0].name == "rq1_learned_freq_temp_head__lr1e-2_formadascale"
    assert variants[3].name == "rq1_learned_freq_temp_head__lr3e-2_formconstant"
    for v, raw in zip(variants, raws):
        path = Path(v.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(raw))
        cfg = load_config(path)
        assert cfg.name == v.name and cfg.output_dir == v.run_dir
        assert cfg.train.steps == 200  # single-value axis: override, not in the name
        assert cfg.train.lr == v.overrides["train.lr"]
        assert cfg.pos["temp"]["form"] == v.overrides["pos.temp.form"]
        assert cfg.pos["temp"]["length_source"] == "tokens"  # untouched base keys survive
        assert cfg.wandb.run_name == v.name
    assert base["train"]["lr"] == 1.0e-3  # base not mutated


def test_expand_rejects_bad_sweeps(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE.read_text())
    with pytest.raises(ValueError, match="Unknown"):
        pl.expand_variants(base, [("train.lrr", ["1", "2"])], tmp_path)
    with pytest.raises(ValueError):
        pl.expand_variants(base, [("pos.granularity", ["head", "pixel"])], tmp_path)
    with pytest.raises(ValueError, match="set by the pipeline"):
        pl.expand_variants(base, [("output_dir", ["a", "b"])], tmp_path)


BASELINES = sorted((ROOT / "configs" / "rq1").glob("[!_]*.yaml"))


@pytest.mark.parametrize("path", [p for p in BASELINES if yaml.safe_load(p.read_text())["pos"]["mode"] in pl.TRAINING_FREE_MODES], ids=lambda p: p.name)
def test_baseline_configs_become_eval_flags(path: Path, tmp_path: Path) -> None:
    """Every shipped training-free config maps onto eval CLI flags with identical settings."""
    from posext.eval.common import pos_from_args
    from posext.eval.passkey import parse_args as passkey_args
    from posext.eval.ppl import parse_args as ppl_args
    from posext.models.positional import PosConfig

    base = yaml.safe_load(path.read_text())
    (v,), _ = pl.expand_variants(base, [], tmp_path)
    assert not v.trainable and "--run" not in v.model_args
    want = PosConfig.from_dict(base["pos"]).to_dict()
    from posext.eval.ruler import parse_args as ruler_args

    parsers = (ppl_args, passkey_args, lambda a: ruler_args(["run", *a, "--split", "test"]))
    for parse in parsers:  # every eval CLI accepts the flags and rebuilds the same pos
        ns = parse(v.model_args)
        assert PosConfig.from_dict(pos_from_args(ns)).to_dict() == want


def test_eval_model_args_values_and_rejection() -> None:
    args = pl.eval_model_args({"mode": "yarn", "yarn": {"factor": 4, "beta_fast": 64, "beta_slow": 2,
                                                        "attention_factor": 1.2}}, "m")
    assert args == ["--model", "m", "--pos-mode", "yarn", "--factor", "4.0", "--beta-fast", "64.0",
                    "--beta-slow", "2.0", "--attention-factor", "1.2"]
    assert pl.eval_model_args({"mode": "pi", "pi_factor": 6}, "m")[-2:] == ["--factor", "6.0"]
    assert pl.eval_model_args({"mode": "none"}, "m") == ["--model", "m", "--pos-mode", "none"]
    with pytest.raises(ValueError, match="not expressible"):  # eval CLIs have no --truncate flag
        pl.eval_model_args({"mode": "yarn", "yarn": {"truncate": False}}, "m")
    with pytest.raises(ValueError, match="not expressible"):
        pl.eval_model_args({"mode": "pi", "native_len": 2048}, "m")


def test_rank_variants_metrics() -> None:
    rows = [
        {"index": 1, "nll": 3.00, "passkey_accuracy": 0.50},
        {"index": 2, "nll": 2.90, "passkey_accuracy": 0.40},  # best PPL
        {"index": 3, "nll": 2.95, "passkey_accuracy": 0.90},  # best passkey, best combined
    ]
    assert [r["index"] for r in pl.rank_variants(rows, "ppl")] == [2, 3, 1]
    assert [r["index"] for r in pl.rank_variants(rows, "passkey")] == [3, 1, 2]
    combined = pl.rank_variants(rows, "combined")
    assert combined[0]["index"] == 3
    # Same formula as the YaRN grid.
    from posext.baselines.yarn_grid import combined_score

    nlls, accs = [r["nll"] for r in rows], [r["passkey_accuracy"] for r in rows]
    mean = lambda xs: sum(xs) / len(xs)
    std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) ** 0.5
    for r in combined:
        ref = combined_score(r["nll"], r["passkey_accuracy"], mean(nlls), std(nlls), mean(accs), std(accs))
        assert r["score"] == pytest.approx(ref)
    # One candidate: z-scores are 0, still selected.
    assert pl.rank_variants(rows[:1], "combined")[0]["score"] == 0.0


def test_submit_chains_jobs_and_is_idempotent(repo: Path) -> None:
    calls: list[list[str]] = []

    def fake_qsub(cmd: list[str]) -> str:
        calls.append(cmd)
        return str(100 + len(calls))

    args = _submit_args("--sweep", "train.lr=1e-2,3e-2", "--name", "t1", "--gpu-type", "A100", "--max-concurrent", "1")
    plan = pl.submit(args, qsub=fake_qsub)
    stages = [c[c.index("-N") + 1] for c in calls]
    assert stages == ["sw-train-t1", "sw-rulerprep-t1", "sw-select-t1", "sw-test-t1", "sw-ruler-t1",
                      "sw-rulermerge-t1"]
    train, prep, select, test, ruler, merge = calls
    assert train[train.index("-t") + 1] == "1-2" and train[train.index("-tc") + 1] == "1"
    assert "gpu_type=A100" in train and "gpus=1" in train
    assert "STAGE=variant" in train[train.index("-v") + 1]
    assert "-hold_jid" not in train and "-hold_jid" not in prep
    assert prep[-1].endswith("ruler_prepare.qsub") and "SPLIT=test" in prep[prep.index("-v") + 1]
    assert select[select.index("-hold_jid") + 1] == "101" and "gpus=1" not in select
    assert test[test.index("-hold_jid") + 1] == "103"
    assert ruler[ruler.index("-hold_jid") + 1] == "103,102"
    assert ruler[ruler.index("-t") + 1] == "1-3"  # one task per length
    assert merge[merge.index("-hold_jid") + 1] == "105" and "gpus=1" not in merge
    assert all(f"POSEXT_ROOT={repo}" in c[c.index("-v") + 1] for c in calls)

    saved = pl.load_plan(repo / plan["sweep_dir"])
    assert saved["jobs"] == {"train": "101", "ruler_prepare": "102", "select": "103", "test": "104", "ruler": "105",
                             "ruler_merge": "106"}
    assert len(list((repo / plan["sweep_dir"] / "configs").glob("*.yaml"))) == 2
    assert (repo / plan["sweep_dir"] / "logs").is_dir()

    # Same command again: allowed (resubmission). Different plan, same name: refused.
    pl.submit(args, qsub=fake_qsub)
    with pytest.raises(SystemExit, match="different plan"):
        pl.submit(_submit_args("--sweep", "train.lr=1e-1,3e-2", "--name", "t1"), qsub=fake_qsub)


def test_submit_skips_prepare_when_ruler_data_ready(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pl, "ruler_test_ready", lambda tasks, lengths: True)
    calls: list[list[str]] = []
    pl.submit(_submit_args("--sweep", "train.lr=1e-2", "--name", "t2"), qsub=lambda c: calls.append(c) or str(len(calls)))
    names = [c[c.index("-N") + 1] for c in calls]
    assert names == ["sw-train-t2", "sw-select-t2", "sw-test-t2", "sw-ruler-t2", "sw-rulermerge-t2"]
    assert calls[3][calls[3].index("-hold_jid") + 1] == "2"


def test_submit_no_ruler_and_dry_run(repo: Path) -> None:
    def no_qsub(cmd: list[str]) -> str:
        raise AssertionError("dry run must not submit")

    plan = pl.submit(_submit_args("--sweep", "train.lr=1e-2,3e-2", "--name", "t3", "--dry-run"), qsub=no_qsub)
    assert not (repo / plan["sweep_dir"]).exists()
    calls: list[list[str]] = []
    pl.submit(_submit_args("--name", "t4", "--no-ruler"), qsub=lambda c: calls.append(c) or str(len(calls)))
    assert [c[c.index("-N") + 1] for c in calls] == ["sw-train-t4", "sw-select-t4", "sw-test-t4"]


def test_ruler_test_ready(repo: Path) -> None:
    from posext.eval.ruler import data_file, default_data_root
    from posext.models.build import DEFAULT_MODEL

    assert not pl.ruler_test_ready("niah_single_1", "4096")
    path = data_file(default_data_root(DEFAULT_MODEL), 4096, "niah_single_1", "test")
    path.parent.mkdir(parents=True)
    path.write_text("")
    meta = {"seed": 42, "split": "test", "num_samples": 500}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    assert pl.ruler_test_ready("niah_single_1", "4096")
    path.with_suffix(".meta.json").write_text(json.dumps({**meta, "num_samples": 20}))
    assert not pl.ruler_test_ready("niah_single_1", "4096")


def _fake_val(run_dir: Path, nll: float, acc: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lengths = {str(L): {"nll": nll, "ppl": 2.718281828 ** nll} for L in (4096, 8192, 16384)}
    (run_dir / "val_ppl.json").write_text(json.dumps({"lengths": lengths}))
    (run_dir / "val_passkey.json").write_text(json.dumps({"accuracy": acc, "by_length": {}}))


def test_stages_end_to_end_with_fake_commands(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = pl.submit(_submit_args("--sweep", "train.lr=1e-2,3e-2,1e-1", "--name", "t5"), qsub=lambda c: "1")
    sweep_dir = Path(plan["sweep_dir"])
    v1, v2, v3 = (Path(v["run_dir"]) for v in plan["variants"])

    ran: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        ran.append(cmd)
        module = cmd[2]
        if module == "posext.train":
            out = Path(load_config(cmd[cmd.index("--config") + 1]).output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "train_summary.json").write_text("{}")
        else:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            fake = {
                "posext.eval.ppl": {"lengths": {"16384": {"nll": 3.0, "ppl": 20.1}}},
                "posext.eval.passkey": {"accuracy": 0.9, "by_length": {"16384": 0.8}},
                "posext.eval.ruler": {"avg": 50.0, "avg_by_length": {"16384": 40.0}, "complete": True},
            }[module]
            out.write_text(json.dumps(fake))

    pl.stage_variant(sweep_dir, 1, run=fake_run)
    assert [c[2] for c in ran] == ["posext.train", "posext.eval.ppl", "posext.eval.passkey"]
    assert "--resume" in ran[0]
    assert ran[1][ran[1].index("--split") + 1] == "validation"
    assert ran[2][ran[2].index("--seed") + 1] == "0"
    ran.clear()
    pl.stage_variant(sweep_dir, 1, run=fake_run)  # everything done: nothing re-runs
    assert ran == []

    # Variant 3 never finished; 2 is best on both metrics.
    _fake_val(v1, 3.0, 0.5)
    _fake_val(v2, 2.9, 0.9)
    shutil.rmtree(v3, ignore_errors=True)
    sel = pl.stage_select(sweep_dir)
    assert sel["best"]["name"] == plan["variants"][1]["name"]
    assert sel["missing"] == [plan["variants"][2]["name"]]

    pl.stage_test(sweep_dir, run=fake_run)
    for i in (1, 2, 3):
        pl.stage_ruler(sweep_dir, i, run=fake_run)
    test_ppl, test_pk, ruler, _, ruler16 = ran
    assert [c[c.index("--lengths") + 1] for c in ran[2:]] == ["4096", "8192", "16384"]
    assert len({c[c.index("--pred-dir") + 1] for c in ran[2:]}) == 1  # one shared prediction dir
    assert ruler16[ruler16.index("--out") + 1].endswith("result_16384.json")
    assert "--max-samples" not in ruler
    assert test_ppl[test_ppl.index("--run") + 1] == str(v2)
    assert test_ppl[test_ppl.index("--split") + 1] == "test"
    assert test_pk[test_pk.index("--seed") + 1] == "1"
    assert ruler[3] == "run" and ruler[ruler.index("--split") + 1] == "test"
    ran.clear()
    pl.stage_test(sweep_dir, run=fake_run)
    for i in (1, 2, 3):
        pl.stage_ruler(sweep_dir, i, run=fake_run)
    assert ran == []
    assert "lengths with results: 16384, 4096, 8192" in pl.status(sweep_dir)

    import posext.eval.ruler as ruler_mod

    def fake_score(pred_dir, *, tasks, lengths):
        assert list(lengths) == [4096, 8192, 16384] and len(tasks) == 8
        return {"avg": 50.0, "avg_by_length": {"16384": 40.0}, "complete": True}

    monkeypatch.setattr(ruler_mod, "score_ruler", fake_score)
    merged = pl.stage_ruler_merge(sweep_dir)
    assert merged["tag"] == sel["best"]["name"] and merged["source"] == ["--run", str(v2)]
    text = pl.status(sweep_dir)
    assert "<- selected" in text and "PPL@16384=20.100" in text and "avg=50.0" in text
    Path(pl.test_paths(plan, sel["best"]["name"])["ppl"]).write_text("{}")  # odd file: no crash
    assert "unexpected format" in pl.status(sweep_dir)


def test_select_fails_when_nothing_finished(repo: Path) -> None:
    plan = pl.submit(_submit_args("--name", "t6"), qsub=lambda c: "1")
    with pytest.raises(SystemExit, match="nothing to select"):
        pl.stage_select(Path(plan["sweep_dir"]))
    assert "Not selected yet" in pl.status(Path(plan["sweep_dir"]))


def test_variant_stage_stops_if_training_unfinished(repo: Path) -> None:
    plan = pl.submit(_submit_args("--name", "t7"), qsub=lambda c: "1")
    with pytest.raises(RuntimeError, match="did not finish"):
        pl.stage_variant(Path(plan["sweep_dir"]), 1, run=lambda cmd: None)


def test_hms() -> None:
    assert pl.hms(6) == "6:00:00"
    assert pl.hms(0.5) == "0:30:00"
    assert pl.hms(24.25) == "24:15:00"


def test_ruler_lists_survive_qsub_v(repo: Path) -> None:
    """Regression (SCC dry run): qsub -v splits on commas, so prepare gets a:b lists."""
    args = _submit_args("--name", "t10", "--ruler-tasks", "niah_single_1,niah_single_2",
                        "--ruler-lengths", "4096,8192", "--dry-run")
    cmds = pl.qsub_commands(pl.build_plan(args)[0], args, prepare_ruler=True)
    env = cmds["ruler_prepare"][cmds["ruler_prepare"].index("-v") + 1]
    assert "TASKS=niah_single_1:niah_single_2" in env and "LENGTHS=4096:8192" in env
    assert env.count(",") == env.count("=") - 1  # commas only between variables
    script = (ROOT / "scripts" / "scc" / "ruler_prepare.qsub").read_text()
    assert '"${TASKS//:/,}"' in script and '"${LENGTHS//:/,}"' in script


def test_ruler_smoke_options_reach_the_ruler_command(repo: Path) -> None:
    args = _submit_args("--name", "t8", "--ruler-tasks", "niah_single_1", "--ruler-lengths", "4096",
                        "--ruler-max-samples", "5")
    calls: list[list[str]] = []
    plan = pl.submit(args, qsub=lambda c: calls.append(c) or str(len(calls)))
    prep_env = calls[1][calls[1].index("-v") + 1]
    assert "TASKS=niah_single_1" in prep_env and "LENGTHS=4096" in prep_env
    sweep_dir = Path(plan["sweep_dir"])
    _fake_val(Path(plan["variants"][0]["run_dir"]), 3.0, 0.5)
    pl.stage_select(sweep_dir)
    ran: list[list[str]] = []
    pl.stage_ruler(sweep_dir, 1, run=ran.append)
    (cmd,) = ran
    assert cmd[cmd.index("--tasks") + 1] == "niah_single_1"
    assert cmd[cmd.index("--lengths") + 1] == "4096"
    assert cmd[cmd.index("--max-samples") + 1] == "5"


def test_baseline_pipeline_skips_training(repo: Path) -> None:
    """A training-free config runs the same stages, with --pos-mode flags instead of --run."""
    args = pl.parse_args(["submit", str(ROOT / "configs" / "rq1" / "yarn_tuned.yaml"), "--name", "t9"])
    calls: list[list[str]] = []
    plan = pl.submit(args, qsub=lambda c: calls.append(c) or str(len(calls)))
    train = calls[0]
    assert train[train.index("-N") + 1] == "sw-train-t9"  # same chain: val evals still run
    assert train[train.index("-l") + 1] == "h_rt=1:00:00"  # nothing to train: short default
    sweep_dir = Path(plan["sweep_dir"])
    ran: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        ran.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")

    pl.stage_variant(sweep_dir, 1, run=fake_run)
    assert [c[2] for c in ran] == ["posext.eval.ppl", "posext.eval.passkey"]  # no posext.train
    for cmd in ran:
        assert "--run" not in cmd
        assert cmd[cmd.index("--pos-mode") + 1] == "yarn"
        assert cmd[cmd.index("--attention-factor") + 1] == "1.2"
    run_dir = Path(plan["variants"][0]["run_dir"])
    _fake_val(run_dir, 3.0, 0.9)
    best = pl.stage_select(sweep_dir)["best"]
    ran.clear()
    pl.stage_test(sweep_dir, run=fake_run)
    pl.stage_ruler(sweep_dir, 3, run=fake_run)
    assert all(c[c.index("--beta-fast") + 1] == "64.0" for c in ran)
    assert ran[-1][3] == "run" and ran[-1][ran[-1].index("--lengths") + 1] == "16384"
    assert best["model_args"] == plan["variants"][0]["model_args"]


def test_old_plans_without_model_args_still_work() -> None:
    assert pl.model_args({"run_dir": "outputs/x"}) == ["--run", "outputs/x"]


def test_combined_tie_goes_to_lower_nll() -> None:
    """Regression (SCC smoke run): two candidates that each win one metric tie exactly; float
    noise used to pick the higher-NLL one."""
    rows = [
        {"index": 1, "nll": 5.9947, "passkey_accuracy": 0.333},
        {"index": 2, "nll": 5.8091, "passkey_accuracy": 0.133},
    ]
    ranked = pl.rank_variants(rows, "combined")
    assert ranked[0]["score"] == ranked[1]["score"] == 0.0
    assert ranked[0]["index"] == 2


def test_status_no_ruler(repo: Path) -> None:
    plan = pl.submit(_submit_args("--name", "t11", "--no-ruler"), qsub=lambda c: "1")
    sweep_dir = Path(plan["sweep_dir"])
    _fake_val(Path(plan["variants"][0]["run_dir"]), 3.0, 0.5)
    pl.stage_select(sweep_dir)
    assert "not run (--no-ruler)" in pl.status(sweep_dir)


# ---- retest: more test passkey prompts for each sweep's selected model ------------------


def _selected_sweep(name: str) -> Path:
    plan = pl.submit(_submit_args("--name", name, "--no-ruler"), qsub=lambda c: str(500 + len(name)))
    sweep_dir = Path(plan["sweep_dir"])
    _fake_val(Path(plan["variants"][0]["run_dir"]), 3.0, 0.5)
    pl.stage_select(sweep_dir)
    return sweep_dir


def test_retest_submits_holds_and_skips(repo: Path) -> None:
    done = _selected_sweep("r_done")  # selected; select job finished
    pending = Path(pl.submit(_submit_args("--name", "r_pending", "--no-ruler"), qsub=lambda c: "777")["sweep_dir"])
    failed = Path(pl.submit(_submit_args("--name", "r_failed", "--no-ruler"), qsub=lambda c: "888")["sweep_dir"])
    calls: list[list[str]] = []
    args = pl.parse_args(["retest", str(done), str(pending), str(failed), "--passkey-n-per-cell", "50"])
    out = pl.retest(args, qsub=lambda c: calls.append(c) or str(900 + len(calls)), queued=lambda j: j == "777")
    assert out == {"r_done": "901", "r_pending": "902"}  # r_failed: no selection, select job gone
    first, second = calls
    assert "-hold_jid" not in first
    assert second[second.index("-hold_jid") + 1] == "777"  # waits for the still-queued select job
    for c in calls:
        env = c[c.index("-v") + 1]
        assert "STAGE=retest" in env and "PASSKEY_N=50" in env and "gpus=1" in c and "h_rt=2:00:00" in c
        assert c[c.index("-N") + 1].startswith("sw-retest50-")
    assert pl.load_plan(pending)["jobs"]  # plan.json untouched (same-plan resubmission still works)
    assert "retest" not in json.dumps(pl.load_plan(done))


def test_retest_dry_run_and_already_done(repo: Path) -> None:
    sweep = _selected_sweep("r_dry")
    args = pl.parse_args(["retest", str(sweep), "--passkey-n-per-cell", "50", "--dry-run"])

    def no_qsub(cmd: list[str]) -> str:
        raise AssertionError("dry run must not submit")

    assert pl.retest(args, qsub=no_qsub, queued=lambda j: False) == {}
    plan = pl.load_plan(sweep)
    best = json.loads((sweep / "selection.json").read_text())["best"]["name"]
    out = pl.retest_path(plan, best, 50)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"accuracy": 0.8, "n": 750, "by_length": {"16384": 0.6}}))
    args = pl.parse_args(["retest", str(sweep), "--passkey-n-per-cell", "50"])
    assert pl.retest(args, qsub=no_qsub, queued=lambda j: False) == {}  # result exists: skipped
    assert "passkey n50: acc=0.800  (n=750)" in pl.status(sweep)


def test_stage_retest_command(repo: Path) -> None:
    sweep = _selected_sweep("r_stage")
    ran: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        ran.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)  # as posext.eval.passkey does
        out.write_text("{}")

    pl.stage_retest(sweep, 50, run=fake_run)
    (cmd,) = ran
    plan = pl.load_plan(sweep)
    assert cmd[2] == "posext.eval.passkey" and cmd[cmd.index("--seed") + 1] == "1"
    assert cmd[cmd.index("--n-per-cell") + 1] == "50" and "--run" in cmd
    assert cmd[cmd.index("--out") + 1].endswith(f"test_passkey_n50_{plan['variants'][0]['name']}.json")
    pl.stage_retest(sweep, 50, run=fake_run)  # already done: nothing re-runs
    assert len(ran) == 1


def test_stage_script_and_wrapper_know_retest() -> None:
    stage = (ROOT / "scripts" / "scc" / "sweep_stage.qsub").read_text()
    assert 'retest_stage "$SWEEP_DIR" --n "${PASSKEY_N:?' in stage
    wrapper = (ROOT / "scripts" / "scc" / "pipeline.sh").read_text()
    assert 'if [[ "${1:-}" == retest ]]' in wrapper
