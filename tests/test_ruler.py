"""RULER wrapper (posext/eval/ruler.py): data generation plumbing, resumable inference, scoring.

Most tests use a fake RULER tree (a stand-in prepare.py plus RULER's two config files), so
they run offline in milliseconds. One opt-in test runs the real RULER checkout:
POSEXT_NETWORK=1 POSEXT_RULER_ROOT=/path/to/RULER uv run pytest -m network tests/test_ruler.py
(after `python -m posext.eval.ruler setup`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from posext.eval import common, ruler
from tests.conftest import make_posext

FAKE_SYNTHETIC_YAML = """
niah_single_1:
  task: niah
  args: {type_haystack: noise, num_needle_k: 1}
niah_single_2:
  task: niah
  args: {type_haystack: essay, num_needle_k: 1}
niah_silent_fail:
  task: niah
  args: {type_haystack: noise}
vt:
  task: variable_tracking
  args: {num_chains: 1}
"""

FAKE_CONSTANTS = """
TASKS = {
    'niah': {'tokens_to_generate': 8, 'template': 'x', 'answer_prefix': ' A:'},
    'variable_tracking': {'tokens_to_generate': 3, 'template': 'x', 'answer_prefix': ' B:'},
}
"""

# Mimics RULER's prepare.py: same CLI, writes <save_dir>/<task>/<subset>.jsonl. Like the real
# one it swallows task-script failures (exit 0, no file) for `niah_silent_fail`.
FAKE_PREPARE = '''
import argparse, json, os, shutil, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ["--save_dir", "--benchmark", "--task", "--subset", "--tokenizer_path",
          "--tokenizer_type", "--model_template_type"]:
    p.add_argument(a)
for a in ["--max_seq_length", "--num_samples", "--random_seed"]:
    p.add_argument(a, type=int)
a = p.parse_args()
assert shutil.which("python") and Path(shutil.which("python")).parent == Path(sys.executable).parent
if a.task == "niah_silent_fail":
    print("Error output: boom")
    sys.exit(0)
out = Path(a.save_dir) / a.task / f"{a.subset}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for i in range(a.num_samples):
        key = str(1000 + a.random_seed + i)
        row = {"index": i, "input": f"needle {key} seed {a.random_seed}.", "outputs": [key],
               "length": 40, "answer_prefix": " The answer is"}
        f.write(json.dumps(row) + "\\n")
with open(Path(a.save_dir) / "calls.log", "a") as f:
    f.write(f"{a.task} {a.subset} {a.random_seed}\\n")
'''


@pytest.fixture
def fake_ruler(tmp_path: Path) -> Path:
    """A minimal RULER tree with the files ruler.py reads or runs."""
    root = tmp_path / "RULER"
    (root / "scripts" / "data" / "synthetic" / "json").mkdir(parents=True)
    (root / "scripts" / "synthetic.yaml").write_text(FAKE_SYNTHETIC_YAML)
    (root / "scripts" / "data" / "synthetic" / "constants.py").write_text(FAKE_CONSTANTS)
    (root / "scripts" / "data" / "prepare.py").write_text(FAKE_PREPARE)
    return root


def _prepare(fake_ruler: Path, data_root: Path, **kw):
    kw = {"tasks": ("niah_single_1",), "lengths": (4096,), "split": "validation",
          "num_samples": 5, "allow_unpinned": True, "require_nltk": False, **kw}
    return ruler.prepare_ruler(ruler_root=fake_ruler, data_root=data_root, **kw)


def _calls(data_root: Path, length: int = 4096) -> list[str]:
    log = data_root / str(length) / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


class CharTokenizer:
    """Character-level stand-in: ids are byte values (< 256 = tiny vocab)."""

    eos_token_id = 0

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) % 256 for c in text]}

    @staticmethod
    def decode(ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


# ------------------------------------------------------------------------- metrics


def test_metrics_match_ruler_definitions() -> None:
    """string_match_all = mean fraction of refs found; part = any ref; case-insensitive; percent."""
    preds = ["The numbers are 123 and 456.", "nothing here", "ABC-def"]
    refs = [["123", "456"], ["123", "789"], ["abc-DEF"]]
    assert ruler.string_match_all(preds, refs) == round((1 + 0 + 1) / 3 * 100, 2)
    assert ruler.string_match_all(["only 123"], [["123", "456"]]) == 50.0
    assert ruler.string_match_part(["only 123"], [["123", "456"]]) == 100.0
    assert ruler.string_match_part(["none"], [["123"]]) == 0.0


def test_postprocess_pred() -> None:
    """Strip, control characters -> newline, strip again (RULER's evaluate.py)."""
    assert ruler.postprocess_pred("  12\x0034\t\n") == "12\n34"
    assert ruler.postprocess_pred("\n\n") == ""


def test_niah_tasks_and_split_settings() -> None:
    """The 8 NIAH variants by default; validation and test use different seeds."""
    assert len(ruler.NIAH_TASKS) == 8 and all(t.startswith("niah_") for t in ruler.NIAH_TASKS)
    assert ruler.DEFAULT_TASKS == ruler.NIAH_TASKS
    assert ruler.SPLIT_SEEDS["validation"] != ruler.SPLIT_SEEDS["test"]
    assert ruler.DEFAULT_NUM_SAMPLES == {"validation": 100, "test": 500}


# ------------------------------------------------------------------------- data generation


def test_task_specs_read_from_ruler(fake_ruler: Path) -> None:
    """Base task, generation budget and args come from RULER's own config files."""
    specs = ruler.task_specs(fake_ruler, ["niah_single_1", "vt"])
    assert specs["niah_single_1"] == {
        "base_task": "niah", "tokens_to_generate": 8,
        "args": {"type_haystack": "noise", "num_needle_k": 1},
    }
    assert specs["vt"]["base_task"] == "variable_tracking"
    assert specs["vt"]["tokens_to_generate"] == 3
    with pytest.raises(ValueError, match="Unknown RULER task"):
        ruler.task_specs(fake_ruler, ["nope"])


def test_prepare_writes_checked_files_and_skips_when_current(fake_ruler: Path, tmp_path: Path) -> None:
    """prepare runs RULER's prepare.py once per file, records metadata, then skips."""
    data_root = tmp_path / "data"
    (path,) = _prepare(fake_ruler, data_root)
    assert path == data_root / "4096" / "niah_single_1" / "validation.jsonl"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["seed"] == ruler.SPLIT_SEEDS["validation"] and meta["num_samples"] == 5
    assert meta["tokens_to_generate"] == 8 and meta["base_task"] == "niah"
    assert meta["model_template_type"] == "base"
    assert len(ruler.read_jsonl(path)) == 5

    _prepare(fake_ruler, data_root)
    assert len(_calls(data_root)) == 1  # skipped: up to date
    _prepare(fake_ruler, data_root, force=True)
    assert len(_calls(data_root)) == 2


def test_prepare_regenerates_stale_file_with_same_count(fake_ruler: Path, tmp_path: Path) -> None:
    """RULER's prepare.py would keep a file with the right line count even from another seed;
    we delete it first when our metadata differs."""
    data_root = tmp_path / "data"
    (path,) = _prepare(fake_ruler, data_root)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["seed"] = 999
    meta_path.write_text(json.dumps(meta))
    _prepare(fake_ruler, data_root)
    assert len(_calls(data_root)) == 2
    assert json.loads(meta_path.read_text())["seed"] == ruler.SPLIT_SEEDS["validation"]


def test_validation_and_test_data_differ(fake_ruler: Path, tmp_path: Path) -> None:
    """Separate files, separate seeds, different prompts."""
    data_root = tmp_path / "data"
    (val,) = _prepare(fake_ruler, data_root, split="validation")
    (test,) = _prepare(fake_ruler, data_root, split="test")
    assert val != test
    v, t = ruler.read_jsonl(val), ruler.read_jsonl(test)
    assert {r["outputs"][0] for r in v}.isdisjoint({r["outputs"][0] for r in t})


def test_prepare_detects_silent_failure(fake_ruler: Path, tmp_path: Path) -> None:
    """prepare.py exits 0 when its task script fails; we check for the file."""
    with pytest.raises(RuntimeError, match=r"(?s)prepare failed.*boom"):
        _prepare(fake_ruler, tmp_path / "data", tasks=("niah_silent_fail",))


def test_prepare_requires_essays_for_essay_haystack(fake_ruler: Path, tmp_path: Path) -> None:
    """Essay-haystack tasks need the downloaded essays (setup step)."""
    with pytest.raises(FileNotFoundError, match="PaulGrahamEssays"):
        _prepare(fake_ruler, tmp_path / "data", tasks=("niah_single_2",))
    (fake_ruler / "scripts/data/synthetic/json/PaulGrahamEssays.json").write_text('{"text": "x"}')
    _prepare(fake_ruler, tmp_path / "data", tasks=("niah_single_2",))


def test_missing_or_unpinned_ruler_root(fake_ruler: Path, tmp_path: Path) -> None:
    """A missing checkout points at `setup`; an unpinned one needs --allow-unpinned."""
    with pytest.raises(FileNotFoundError, match="setup"):
        ruler.check_ruler_root(tmp_path / "nowhere")
    with pytest.raises(RuntimeError, match="pinned"):
        ruler.check_ruler_root(fake_ruler)
    assert ruler.check_ruler_root(fake_ruler, allow_unpinned=True) == "unknown"


def test_validate_samples_errors(tmp_path: Path) -> None:
    """Wrong count, over-long sample, empty reference are all rejected."""
    good = [{"index": 0, "input": "x", "outputs": ["1"], "length": 10}]
    ruler.validate_samples(good, num_samples=1, length=16, path=tmp_path)
    with pytest.raises(RuntimeError, match="expected 2"):
        ruler.validate_samples(good, num_samples=2, length=16, path=tmp_path)
    with pytest.raises(RuntimeError, match="> 8"):
        ruler.validate_samples(good, num_samples=1, length=8, path=tmp_path)
    with pytest.raises(RuntimeError, match="empty references"):
        ruler.validate_samples([{**good[0], "outputs": [""]}], num_samples=1, length=16, path=tmp_path)


def test_load_rejects_wrong_split_metadata(fake_ruler: Path, tmp_path: Path) -> None:
    """Data files carry their seed; loading checks it against the split."""
    data_root = tmp_path / "data"
    (path,) = _prepare(fake_ruler, data_root)
    with pytest.raises(FileNotFoundError, match="prepare --split test"):
        ruler.load_ruler_data(data_root, "niah_single_1", 4096, "test")
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    path.with_suffix(".meta.json").write_text(json.dumps({**meta, "seed": 7}))
    with pytest.raises(RuntimeError, match="does not match"):
        ruler.load_ruler_data(data_root, "niah_single_1", 4096, "validation")


# ------------------------------------------------------------------------- inference + scoring


def _fake_generate(answer_every: int = 2):
    """greedy_generate stand-in: answers correctly on every `answer_every`-th call."""
    calls: list[list[int]] = []

    def gen(model, input_ids, *, max_new_tokens, pad_token_id):
        calls.append(list(input_ids))
        prompt = "".join(chr(i) for i in input_ids)
        key = prompt.split("needle ")[1].split(" ")[0]
        text = f" {key}." if len(calls) % answer_every == 1 or answer_every == 1 else " no idea"
        return [ord(c) for c in text][:max_new_tokens]

    return gen, calls


def test_run_and_score(fake_ruler: Path, tmp_path: Path, monkeypatch) -> None:
    """Prompt = input + answer_prefix; budget from RULER constants; scores per task and length."""
    data_root, pred_dir = tmp_path / "data", tmp_path / "pred"
    _prepare(fake_ruler, data_root, tasks=("niah_single_1", "vt"), lengths=(4096, 8192), num_samples=4)
    gen, calls = _fake_generate(answer_every=1)
    monkeypatch.setattr(common, "greedy_generate", gen)
    kw = {"tasks": ("niah_single_1", "vt"), "lengths": (4096, 8192)}
    ruler.run_ruler(_NoModel(), CharTokenizer(), data_root=data_root,
                    pred_dir=pred_dir, split="validation", **kw)
    assert len(calls) == 16
    assert "".join(chr(i) for i in calls[0]).endswith(". The answer is")
    rows = ruler.read_jsonl(pred_dir / "4096" / "niah_single_1.jsonl")
    assert len(rows) == 4 and all(r["score"] == 100.0 for r in rows)
    # vt has tokens_to_generate = 3: predictions are cut to 3 characters (" 12"), so no match.
    vt = ruler.read_jsonl(pred_dir / "4096" / "vt.jsonl")
    assert all(len(r["pred"]) == 3 and r["score"] == 0.0 for r in vt)

    res = ruler.score_ruler(pred_dir, **kw)
    assert res["scores"]["4096"] == {"niah_single_1": 100.0, "vt": 0.0}
    assert res["avg_by_length"] == {"4096": 50.0, "8192": 50.0}
    assert res["avg"] == 50.0 and res["complete"] is True
    assert res["split"] == "validation" and res["seed"] == [ruler.SPLIT_SEEDS["validation"]]
    assert res["n"]["8192"]["vt"] == 4


class _NoModel:
    """run_ruler only calls .eval() itself; generation is monkeypatched."""

    def eval(self):
        return self


def test_run_resumes_after_interruption(fake_ruler: Path, tmp_path: Path, monkeypatch) -> None:
    """Rows already predicted are skipped; a truncated last line is dropped and redone."""
    data_root, pred_dir = tmp_path / "data", tmp_path / "pred"
    _prepare(fake_ruler, data_root, num_samples=6)
    gen, calls = _fake_generate()
    monkeypatch.setattr(common, "greedy_generate", gen)
    kw = {"data_root": data_root, "pred_dir": pred_dir, "tasks": ("niah_single_1",), "lengths": (4096,)}
    ruler.run_ruler(_NoModel(), CharTokenizer(), **kw)
    assert len(calls) == 6

    pf = pred_dir / "4096" / "niah_single_1.jsonl"
    lines = pf.read_text().splitlines()
    pf.write_text("\n".join(lines[:3]) + "\n" + lines[3][:10])  # killed mid-write of row 3
    assert not ruler.score_ruler(pred_dir, tasks=("niah_single_1",), lengths=(4096,))["complete"]
    ruler.run_ruler(_NoModel(), CharTokenizer(), **kw)
    assert len(calls) == 6 + 3
    rows = ruler.read_jsonl(pf)
    assert [r["index"] for r in rows] == list(range(6))
    assert ruler.score_ruler(pred_dir, tasks=("niah_single_1",), lengths=(4096,))["complete"]


def test_run_refuses_mismatched_predictions(fake_ruler: Path, tmp_path: Path, monkeypatch) -> None:
    """Predictions from other data/settings are not mixed in; --overwrite starts over."""
    data_root, pred_dir = tmp_path / "data", tmp_path / "pred"
    _prepare(fake_ruler, data_root, num_samples=4)
    gen, _ = _fake_generate()
    monkeypatch.setattr(common, "greedy_generate", gen)
    kw = {"data_root": data_root, "pred_dir": pred_dir, "tasks": ("niah_single_1",), "lengths": (4096,)}
    ruler.run_ruler(_NoModel(), CharTokenizer(), max_samples=2, **kw)
    with pytest.raises(RuntimeError, match="different data"):
        ruler.run_ruler(_NoModel(), CharTokenizer(), max_samples=3, **kw)
    ruler.run_ruler(_NoModel(), CharTokenizer(), max_samples=3, overwrite=True, **kw)
    assert len(ruler.read_jsonl(pred_dir / "4096" / "niah_single_1.jsonl")) == 3


def test_score_flags_missing_cells(fake_ruler: Path, tmp_path: Path, monkeypatch) -> None:
    """Missing (task, length) cells: complete=False, no average for that length."""
    data_root, pred_dir = tmp_path / "data", tmp_path / "pred"
    _prepare(fake_ruler, data_root, num_samples=2)
    monkeypatch.setattr(common, "greedy_generate", _fake_generate(1)[0])
    ruler.run_ruler(_NoModel(), CharTokenizer(), data_root=data_root, pred_dir=pred_dir,
                    tasks=("niah_single_1",), lengths=(4096,))
    res = ruler.score_ruler(pred_dir, tasks=("niah_single_1", "vt"), lengths=(4096,))
    assert res["complete"] is False and res["avg_by_length"] == {} and res["avg"] is None


def test_run_with_tiny_model(fake_ruler: Path, tmp_path: Path) -> None:
    """End to end through the real greedy_generate on a tiny posext model."""
    data_root, pred_dir = tmp_path / "data", tmp_path / "pred"
    _prepare(fake_ruler, data_root, num_samples=2)
    model = make_posext({"mode": "learned_freq", "granularity": "head"})
    ruler.run_ruler(model, CharTokenizer(), data_root=data_root, pred_dir=pred_dir,
                    tasks=("niah_single_1",), lengths=(4096,))
    rows = ruler.read_jsonl(pred_dir / "4096" / "niah_single_1.jsonl")
    assert len(rows) == 2
    assert all(len(r["pred"]) <= 8 and r["prompt_tokens"] > 0 for r in rows)


def test_cli_parses_model_and_paths() -> None:
    """`run` accepts the shared model flags; tag defaults to run dir name or pos mode."""
    args = ruler.parse_args(["run", "--pos-mode", "yarn", "--factor", "4", "--beta-fast", "16",
                             "--attention-factor", "1.1", "--split", "test", "--lengths", "4096"])
    assert args.cmd == "run" and args.split == "test" and args.attention_factor == 1.1
    assert ruler._tag(args) == "yarn"
    assert common.pos_from_args(args)["yarn"]["beta_fast"] == 16
    args = ruler.parse_args(["run", "--run", "outputs/rq1_learned_freq_head"])
    assert ruler._tag(args) == "rq1_learned_freq_head"
    assert ruler.parse_args(["score", "--pred-dir", "x"]).cmd == "score"


# ------------------------------------------------------------------------- real RULER (opt-in)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("POSEXT_NETWORK") != "1", reason="set POSEXT_NETWORK=1")
def test_real_ruler_prepare(tmp_path: Path) -> None:
    """RULER's real scripts with the OLMo-2 tokenizer: prompts fit, answers are in the input."""
    from transformers import AutoTokenizer

    root = ruler.default_ruler_root()
    if not (root / "scripts" / "data" / "prepare.py").is_file():
        pytest.skip(f"no RULER checkout at {root}; run `python -m posext.eval.ruler setup`")
    tasks = ("niah_single_1", "niah_multikey_2", "niah_multivalue")
    paths = ruler.prepare_ruler(ruler_root=root, data_root=tmp_path, tasks=tasks,
                                lengths=(4096,), split="validation", num_samples=3)
    tok = AutoTokenizer.from_pretrained(ruler.DEFAULT_MODEL)
    specs = ruler.task_specs(root, tasks)
    for task, path in zip(tasks, paths):
        for row in ruler.read_jsonl(path):
            prompt = row["input"] + row["answer_prefix"]
            n = len(tok(prompt)["input_ids"])
            assert n + specs[task]["tokens_to_generate"] <= 4096
            assert n > 3000, n  # haystack actually fills the context
            assert all(ref in row["input"] for ref in row["outputs"])
