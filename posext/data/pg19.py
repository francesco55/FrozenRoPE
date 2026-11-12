"""PG-19 books -> fixed-length token sequences (non-overlapping, within one book), cached on disk.

Data source: the `emozilla/pg19` Parquet mirror of PG-19 (Rae et al., 2020). The official
`deepmind/pg19` uses a loading script that `datasets` >= 4 no longer runs. The mirror's
validation (50 books) and test (100 books) splits are complete; its train split has 13,684 of
the 28,602 books, which is far more than positional-only training needs.

Split discipline (CLAUDE.md): `train` for training, `validation` for the YaRN grid search and
hyperparameters, `test` only for final numbers.

Typical use:

    chunks = load_pg19_chunks("test", seq_len=16384, max_sequences=50)   # (N, 16384) int64
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "emozilla/pg19"
DEFAULT_TOKENIZER = "allenai/OLMo-2-0425-1B"
SPLITS = ("train", "validation", "test")
EXPECTED_BOOKS = {"validation": 50, "test": 100}  # complete in the mirror; checked on load
CACHE_ENV = "POSEXT_DATA_CACHE"  # on SCC, point this at /projectnb/<project>/...

Encode = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class ChunkSpec:
    """Everything that determines a set of chunks (and its cache key)."""

    split: str
    seq_len: int
    max_sequences: int | None = None  # None -> every chunk of every book
    max_chunks_per_book: int | None = None  # cap per book, for more diverse training data
    seed: int = 0  # book order
    tokenizer: str = DEFAULT_TOKENIZER
    dataset: str = DEFAULT_DATASET

    def __post_init__(self) -> None:
        """Validate the split and lengths."""
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {self.seq_len}")

    def cache_name(self) -> str:
        """Readable, collision-safe file name for this spec."""
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"pg19_{self.split}_{self.seq_len}_{digest}.pt"


# ---- chunking (pure functions, tested offline) ---------------------------------------


def chunk_tokens(token_ids: Sequence[int], seq_len: int, max_chunks: int | None = None) -> list[list[int]]:
    """Split one book's tokens into non-overlapping seq_len pieces; the leftover tail is dropped."""
    n = len(token_ids) // seq_len
    if max_chunks is not None:
        n = min(n, max_chunks)
    return [list(token_ids[i * seq_len : (i + 1) * seq_len]) for i in range(n)]


def book_order(n_books: int, seed: int) -> list[int]:
    """Deterministic shuffled order of book indices."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_books, generator=generator).tolist()


def iter_chunks(
    texts: Sequence[str], encode: Encode, spec: ChunkSpec
) -> Iterator[tuple[int, list[int]]]:
    """Yield (book_index, chunk) in shuffled book order, tokenizing books only as needed."""
    produced = 0
    for book in book_order(len(texts), spec.seed):
        for chunk in chunk_tokens(encode(texts[book]), spec.seq_len, spec.max_chunks_per_book):
            yield book, chunk
            produced += 1
            if spec.max_sequences is not None and produced >= spec.max_sequences:
                return


def build_chunks(texts: Sequence[str], encode: Encode, spec: ChunkSpec) -> tuple[torch.Tensor, list[int]]:
    """Chunks as a (N, seq_len) int64 tensor, plus the source book index of each row."""
    books, rows = [], []
    for book, chunk in iter_chunks(texts, encode, spec):
        books.append(book)
        rows.append(chunk)
    if not rows:
        return torch.empty(0, spec.seq_len, dtype=torch.long), []
    return torch.tensor(rows, dtype=torch.long), books


# ---- loading with Hugging Face datasets / tokenizers ---------------------------------


def hf_encoder(tokenizer_name: str = DEFAULT_TOKENIZER) -> Encode:
    """Tokenize with the OLMo-2 tokenizer, without special tokens (OLMo-2 adds none anyway)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = 10**12  # books are long; silence the length warning
    return lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]


class _TextColumn(Sequence[str]):
    """Lazy view of a datasets.Dataset's `text` column (avoids loading every book into memory)."""

    def __init__(self, dataset) -> None:
        """Wrap a Hugging Face dataset that has a `text` column."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Number of books."""
        return len(self.dataset)

    def __getitem__(self, index):  # type: ignore[override]
        """Text of one book."""
        return self.dataset[index]["text"]


def split_data_files(split: str) -> dict[str, str]:
    """Parquet files of one split in the mirror's layout (data/<split>-NNNNN-of-NNNNN-*.parquet)."""
    return {split: f"data/{split}-*.parquet"}


def load_pg19_texts(split: str, dataset: str = DEFAULT_DATASET) -> Sequence[str]:
    """PG-19 book texts for a split, checking the complete splits' sizes.

    Only the requested split's files are downloaded. Without `data_files`, `load_dataset`
    downloads and converts every split (~18 GB on disk for the 20 MB test split).
    """
    from datasets import load_dataset

    # no_checks: the repo metadata lists all three splits, and we only build one. We check the
    # book counts of the complete splits ourselves below.
    ds = load_dataset(dataset, data_files=split_data_files(split), split=split, verification_mode="no_checks")
    expected = EXPECTED_BOOKS.get(split)
    if expected is not None and len(ds) != expected:
        raise ValueError(f"PG-19 {split} should have {expected} books, {dataset} has {len(ds)}")
    return _TextColumn(ds)


def default_cache_dir() -> Path:
    """$POSEXT_DATA_CACHE if set, else ./data_cache (gitignored)."""
    return Path(os.environ.get(CACHE_ENV, "data_cache"))


def load_pg19_chunks(
    split: str,
    seq_len: int,
    *,
    max_sequences: int | None = None,
    max_chunks_per_book: int | None = None,
    seed: int = 0,
    tokenizer: str = DEFAULT_TOKENIZER,
    dataset: str = DEFAULT_DATASET,
    cache_dir: str | Path | None = None,
    texts: Sequence[str] | None = None,
    encode: Encode | None = None,
) -> torch.Tensor:
    """PG-19 chunks for a split and length, (N, seq_len) int64, built once and cached.

    `texts` and `encode` override the Hugging Face dataset and tokenizer (used by tests).
    """
    spec = ChunkSpec(split, seq_len, max_sequences, max_chunks_per_book, seed, tokenizer, dataset)
    path = Path(cache_dir or default_cache_dir()) / spec.cache_name()
    if path.exists():
        saved = torch.load(path)
        if saved["spec"] != asdict(spec):  # hash collision or edited file
            raise ValueError(f"Cache file {path} does not match the requested spec")
        return saved["input_ids"].long()

    texts = texts if texts is not None else load_pg19_texts(split, dataset)
    encode = encode or hf_encoder(tokenizer)
    chunks, books = build_chunks(texts, encode, spec)
    if spec.max_sequences is not None and len(chunks) < spec.max_sequences:
        logger.warning("Only %d chunks of length %d available in %s (asked for %d)",
                       len(chunks), seq_len, split, spec.max_sequences)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"spec": asdict(spec), "input_ids": chunks.to(torch.int32), "books": books}, tmp)
    tmp.replace(path)  # atomic: concurrent jobs never read a half-written file
    logger.info("Cached %d PG-19 %s chunks of length %d at %s", len(chunks), split, seq_len, path)
    return chunks
