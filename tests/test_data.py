"""Tests for posext/data: PoSE position ids, PG-19 chunking and caching, training batches.

Offline: a fake dataset and tokenizer stand in for PG-19 and OLMo-2's tokenizer. The real data
test is opt-in (POSEXT_NETWORK=1).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import Olmo2ForCausalLM

from posext.data import (
    ChunkSpec,
    Collator,
    build_chunks,
    chunk_tokens,
    load_pg19_chunks,
    pose_position_ids,
)

from .conftest import make_posext

# ---- PoSE position ids ---------------------------------------------------------------


def jumps(positions: torch.Tensor) -> torch.Tensor:
    """Token indices where a new piece starts (position gap > 1)."""
    return (positions.diff() > 1).nonzero().flatten() + 1


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 4])
def test_pose_invariants(n_chunks: int) -> None:
    """First id 0, strictly increasing, last < L', right length, at most n_chunks contiguous pieces."""
    generator = torch.Generator().manual_seed(0)
    seq_len, target = 64, 256
    for _ in range(300):
        pos = pose_position_ids(seq_len, target, n_chunks, generator)
        assert pos.shape == (seq_len,) and pos.dtype == torch.long
        assert pos[0] == 0
        assert (pos.diff() >= 1).all()
        assert pos.max() < target
        assert len(jumps(pos)) <= n_chunks - 1


def test_pose_two_chunks_matches_reference_scheme() -> None:
    """n_chunks=2 (PoSE): first piece length in [1, ceil(seq_len/2)], skip in [0, L' - seq_len]."""
    generator = torch.Generator().manual_seed(1)
    seq_len, target = 64, 256
    first_lengths, skips = [], []
    for _ in range(2000):
        pos = pose_position_ids(seq_len, target, 2, generator)
        cut = jumps(pos)
        if len(cut):
            first_lengths.append(int(cut[0]))
            skips.append(int(pos[cut[0]] - cut[0]))
    assert 1 <= min(first_lengths) and max(first_lengths) <= (seq_len + 1) // 2
    assert 0 < min(skips) and max(skips) <= target - seq_len
    # Coverage: the largest distance seen gets close to L' - 1 (the point of PoSE).
    assert max(skips) + seq_len - 1 >= 0.95 * (target - 1)


def test_pose_is_reproducible_from_seed() -> None:
    """Same seed, same positions; different seed, different positions."""
    def sample(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.stack([pose_position_ids(4096, 16384, 2, generator) for _ in range(8)])

    assert torch.equal(sample(0), sample(0))
    assert not torch.equal(sample(0), sample(1))


def test_pose_no_budget_is_contiguous() -> None:
    """target_len == seq_len leaves nothing to skip."""
    assert torch.equal(pose_position_ids(32, 32, 2), torch.arange(32))


@pytest.mark.parametrize("args", [(32, 16, 2), (32, 64, 0), (4, 64, 5)])
def test_pose_rejects_bad_arguments(args: tuple) -> None:
    """target_len < seq_len, n_chunks < 1 or > seq_len are errors."""
    with pytest.raises(ValueError):
        pose_position_ids(*args)


# ---- batches -------------------------------------------------------------------------


def test_long_batches() -> None:
    """long regime: position ids 0..L-1, labels = inputs, all-ones attention mask."""
    rows = torch.randint(0, 100, (3, 32))
    batch = Collator("long", target_len=32)(list(rows))
    assert torch.equal(batch["input_ids"], rows) and torch.equal(batch["labels"], rows)
    assert torch.equal(batch["position_ids"], torch.arange(32).expand(3, -1))
    assert torch.equal(batch["attention_mask"], torch.ones(3, 32, dtype=torch.long))


def test_pose_batches() -> None:
    """pose regime: valid PoSE ids, resampled per example, reproducible from the seed, mask passed."""
    rows = torch.randint(0, 100, (8, 64))
    batch = Collator("pose", target_len=1024, seed=3)(list(rows))
    pos = batch["position_ids"]
    assert pos.shape == (8, 64) and (pos[:, 0] == 0).all() and (pos.diff() >= 1).all() and (pos < 1024).all()
    assert len({tuple(r.tolist()) for r in pos}) > 1  # resampled per example
    assert torch.equal(pos, Collator("pose", target_len=1024, seed=3)(list(rows))["position_ids"])
    assert torch.equal(batch["attention_mask"], torch.ones(8, 64, dtype=torch.long))


def test_collator_rejects_bad_settings() -> None:
    """Unknown regime, or long sequences longer than the target, are errors."""
    with pytest.raises(ValueError):
        Collator("short", target_len=16)
    with pytest.raises(ValueError):
        Collator("long", target_len=16)([torch.zeros(32, dtype=torch.long)])


@torch.no_grad()
def test_pose_batch_trains_across_the_skip(stock: Olmo2ForCausalLM) -> None:
    """End to end in training mode: with a PoSE batch, chunk 2's logits depend on chunk 1's tokens."""
    model = make_posext({"mode": "learned_freq"}, stock.state_dict()).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    collator = Collator("pose", target_len=256, seed=0)
    ids = torch.randint(0, 256, (1, 32))
    batch = collator([ids[0]])
    while len(jumps(batch["position_ids"][0])) == 0:  # make sure this sample has a real skip
        batch = collator([ids[0]])
    cut = int(jumps(batch["position_ids"][0])[0])
    other = dict(batch, input_ids=batch["input_ids"].clone())
    other["input_ids"][0, :cut] = (other["input_ids"][0, :cut] + 1) % 256
    out_a = model(**{k: v for k, v in batch.items() if k != "labels"}).logits[0, cut:]
    out_b = model(**{k: v for k, v in other.items() if k != "labels"}).logits[0, cut:]
    assert (out_a - out_b).abs().max() > 1e-6


# ---- PG-19 chunking ------------------------------------------------------------------

# Fake corpus: book b is the token sequence b*1000, b*1000+1, ..., so each token reveals its book.
BOOK_LENGTHS = [25, 7, 40, 16, 33, 9, 50]
TEXTS = [" ".join(str(b * 1000 + i) for i in range(n)) for b, n in enumerate(BOOK_LENGTHS)]


def fake_encode(text: str) -> list[int]:
    """Whitespace 'tokenizer' for the fake corpus."""
    return [int(t) for t in text.split()]


def test_chunk_tokens_non_overlapping_and_drops_tail() -> None:
    """Exact-length, consecutive, non-overlapping pieces; the tail shorter than seq_len is dropped."""
    tokens = list(range(23))
    chunks = chunk_tokens(tokens, 5)
    assert chunks == [list(range(i, i + 5)) for i in range(0, 20, 5)]
    assert chunk_tokens(tokens, 5, max_chunks=2) == chunks[:2]
    assert chunk_tokens(list(range(4)), 5) == []


def test_build_chunks_stays_within_books() -> None:
    """Every row comes from one book, in order, and the reported book index is right."""
    chunks, books = build_chunks(TEXTS, fake_encode, ChunkSpec("train", 8))
    assert chunks.shape == (sum(n // 8 for n in BOOK_LENGTHS), 8)
    for row, book in zip(chunks, books, strict=True):
        assert (row // 1000 == book).all()  # one book per row
        assert (row.diff() == 1).all()  # contiguous text


def test_build_chunks_limits_and_seed() -> None:
    """max_sequences and max_chunks_per_book are respected; the seed sets the book order."""
    spec = ChunkSpec("train", 8, max_sequences=5, seed=0)
    chunks, books = build_chunks(TEXTS, fake_encode, spec)
    assert chunks.shape == (5, 8)
    assert torch.equal(chunks, build_chunks(TEXTS, fake_encode, spec)[0])
    _, other_books = build_chunks(TEXTS, fake_encode, ChunkSpec("train", 8, max_sequences=5, seed=1))
    assert books != other_books
    _, capped = build_chunks(TEXTS, fake_encode, ChunkSpec("train", 8, max_chunks_per_book=1))
    assert len(capped) == len(set(capped)) == sum(n >= 8 for n in BOOK_LENGTHS)


def test_chunk_spec_rejects_unknown_split() -> None:
    """Only train / validation / test exist."""
    with pytest.raises(ValueError):
        ChunkSpec("dev", 8)


def test_load_pg19_chunks_caches(tmp_path: Path) -> None:
    """First call tokenizes and caches; a repeat call reads the cache; a different spec rebuilds."""
    calls = []

    def counting_encode(text: str) -> list[int]:
        calls.append(text)
        return fake_encode(text)

    kwargs = dict(cache_dir=tmp_path, texts=TEXTS, encode=counting_encode, tokenizer="fake", dataset="fake")
    first = load_pg19_chunks("validation", 8, **kwargs)
    n_calls = len(calls)
    assert n_calls > 0 and len(list(tmp_path.iterdir())) == 1
    second = load_pg19_chunks("validation", 8, **kwargs)
    assert torch.equal(first, second) and second.dtype == torch.long
    assert len(calls) == n_calls  # served from cache
    load_pg19_chunks("validation", 16, **kwargs)
    assert len(list(tmp_path.iterdir())) == 2


# ---- real data (opt-in) --------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("POSEXT_NETWORK") != "1", reason="set POSEXT_NETWORK=1")
def test_real_pg19_test_split(tmp_path: Path) -> None:
    """Downloads the 100-book test split (~20 MB) and the OLMo-2 tokenizer; chunks at 4096."""
    from transformers import AutoTokenizer

    chunks = load_pg19_chunks("test", 4096, max_sequences=8, cache_dir=tmp_path)
    assert chunks.shape == (8, 4096)
    text = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B").decode(chunks[0, :50])
    assert len(text.split()) > 10  # decodes back to English text


def test_load_pg19_texts_requests_only_one_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the requested split's files are fetched (a plain load_dataset downloads all ~18 GB)."""
    import datasets

    seen = {}

    def fake_load_dataset(name, **kwargs):
        seen.update(kwargs, name=name)
        return datasets.Dataset.from_dict({"text": ["a book"] * 50})

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    from posext.data.pg19 import load_pg19_texts

    texts = load_pg19_texts("validation")
    assert seen["data_files"] == {"validation": "data/validation-*.parquet"}
    assert seen["split"] == "validation" and len(texts) == 50 and texts[0] == "a book"
    with pytest.raises(ValueError, match="should have 100 books"):
        load_pg19_texts("test")  # the fake returns 50 books
