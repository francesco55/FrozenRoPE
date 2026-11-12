"""CPU tests for RQ1 training/eval plumbing (no real checkpoint)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from posext.config import ExperimentConfig, load_config
from posext.eval.passkey import PasskeyPrompt, answer_passkey, generate_passkey_prompts, key_in_text
from posext.loss import chunked_lm_cross_entropy
from posext.models import freeze_backbone, log_parameter_counts
from posext.train import build_optimizer, clip_gradients, set_seed
from tests.conftest import TINY, make_posext


ROOT = Path(__file__).resolve().parents[1]
RQ1_CONFIGS = sorted((ROOT / "configs" / "rq1").glob("*.yaml"))
SMOKE = ROOT / "configs" / "smoke.yaml"


@pytest.mark.parametrize("path", [SMOKE, *RQ1_CONFIGS], ids=lambda p: p.name)
def test_configs_load(path: Path) -> None:
    """Every shipped YAML must parse and validate."""
    if path.name.startswith("_"):
        pytest.skip("partial base file")
    cfg = load_config(path)
    assert cfg.name
    assert cfg.pos.get("mode")


def test_smoke_config_is_trainable() -> None:
    """Smoke config must have trainable params so train.py can run."""
    cfg = load_config(SMOKE)
    assert cfg.pos["mode"] == "learned_freq"
    assert cfg.train.steps >= 1
    assert cfg.data.seq_len <= cfg.data.target_len


def test_chunked_ce_matches_full() -> None:
    """Chunked CE equals a single full cross-entropy on a tiny head."""
    torch.manual_seed(0)
    vocab, hidden, seq = 32, 16, 40
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    h = torch.randn(2, seq, hidden)
    y = torch.randint(0, vocab, (2, seq))
    full = chunked_lm_cross_entropy(lm_head, h, y, chunk_size=10_000)
    chunked = chunked_lm_cross_entropy(lm_head, h, y, chunk_size=7)
    assert torch.allclose(full, chunked, rtol=1e-5, atol=1e-6)


def test_train_helpers_on_tiny_model() -> None:
    """Optimizer + clip see only positional params."""
    set_seed(0)
    model = make_posext({"mode": "learned_freq", "granularity": "head"})
    freeze_backbone(model)
    counts = log_parameter_counts(model)
    assert counts["trainable"] > 0
    cfg = ExperimentConfig(
        name="tiny",
        pos={"mode": "learned_freq", "granularity": "head"},
        train={"steps": 1, "lr": 1e-3, "grad_clip": 1.0},
    )
    opt = build_optimizer(model, cfg)
    loss = model(
        input_ids=torch.randint(0, TINY["vocab_size"], (1, 16)),
        labels=torch.randint(0, TINY["vocab_size"], (1, 16)),
    ).loss
    loss.backward()
    norm = clip_gradients(model, cfg)
    assert norm >= 0.0
    opt.step()


def test_passkey_prompt_lengths() -> None:
    """Generated prompts have the requested token length and a 5-digit key."""
    # Avoid downloading: use a blank ByteLevel-ish via the OLMo tokenizer only if cached;
    # otherwise a minimal whitespace tokenizer stand-in.
    try:
        tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B", local_files_only=True)
    except Exception:
        pytest.skip("OLMo-2 tokenizer not cached; passkey length test needs it offline")
    prompts = generate_passkey_prompts(tok, lengths=(128, 256), depths=(0.0, 1.0), n_per_cell=1, seed=0)
    assert len(prompts) == 4
    for p in prompts:
        assert len(p.input_ids) == p.length
        assert len(p.key) == 5 and p.key.isdigit()


def test_passkey_generation_matches_full_forward() -> None:
    """Cached greedy generation in answer_passkey = greedy decoding with full forwards (positions 0..n-1).

    Regression: passing position_ids to generate() froze every new token at the last prompt position.
    """
    torch.manual_seed(0)
    model = make_posext({"mode": "learned_freq", "granularity": "head"})
    with torch.no_grad():
        model.posext.a.normal_(0, 0.5)  # non-identity, so wrong positions change the output
    prompt = torch.randint(0, TINY["vocab_size"], (24,)).tolist()

    class _Tok:  # answer_passkey only needs eos_token_id and decode
        eos_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=True):  # noqa: ARG004
            return " ".join(str(int(i)) for i in ids)

    text = answer_passkey(model, _Tok(), PasskeyPrompt(prompt, "00000", len(prompt), 0.0), max_new_tokens=6)

    ids = torch.tensor([prompt])
    with torch.no_grad():
        for _ in range(6):
            next_id = model(input_ids=ids).logits[0, -1].argmax()
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
    assert text == _Tok.decode(ids[0, len(prompt) :])


def test_key_in_text() -> None:
    """Whole-number match for the passkey."""
    assert key_in_text("01234", "The pass key is 01234.")
    assert not key_in_text("01234", "The pass key is 012345.")


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    """Typos in YAML fail loudly."""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"name": "x", "typo_field": 1, "pos": {"mode": "none"}}))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
