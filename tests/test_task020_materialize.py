"""Contract checks for frozen TASK-020 window selection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.task020_materialize import (
    REVISION,
    ordered_shards,
    selected_offsets,
    text_chunks,
)


def test_window_rank_and_no_overlap() -> None:
    """Check exact selector bytes, rank order, and disjoint 512-token windows."""
    ranked = selected_offsets("validation", 40 * 512 + 9)
    assert len(ranked) == 32
    assert len({offset for _, offset in ranked}) == 32
    assert all(offset % 512 == 0 and offset < 40 * 512 for _, offset in ranked)
    assert ranked == sorted(ranked, key=lambda item: (bytes.fromhex(item[0]), item[1]))
    for digest, offset in ranked:
        raw = f"qw38-task020-validation-v1\n{REVISION}\n{offset}".encode()
        assert digest == hashlib.sha256(raw).hexdigest()


def test_incomplete_shard_sequence_rejected(tmp_path: Path) -> None:
    """Reject a split with a missing middle shard before reading its rows."""
    (tmp_path / "train-00000-of-00002.parquet").touch()
    with pytest.raises(ValueError, match="incomplete"):
        ordered_shards(tmp_path, "train")


def test_chunked_tokenization_matches_whole_text(tmp_path: Path) -> None:
    """Verify LF boundaries preserve the frozen tokenizer's exact token IDs."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        ".cache/authorities/qwen3.8-27b-transformers/tokenizer.json"
    )
    text = (
        "Some words, numbers 123 and punctuation!\n"
        " Another line with café, é and Qwen tokens.\n"
        "\n"
        " A third line.\n"
        " Fourth line has <|im_start|> as a special token.\n"
        "Final line"
    )
    path = tmp_path / "joined.txt"
    path.write_text(text, encoding="utf-8")
    chunks = list(text_chunks(path, limit=8))
    assert len(chunks) >= 3
    assert "".join(chunks) == text
    whole = tokenizer.encode(text, add_special_tokens=False).ids
    segmented = [
        token
        for chunk in chunks
        for token in tokenizer.encode(chunk, add_special_tokens=False).ids
    ]
    assert segmented == whole
