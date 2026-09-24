#!/usr/bin/env python3
"""Freeze WikiText token windows for TASK-020 calibration and screening."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import struct
from collections.abc import Iterator
from pathlib import Path

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
WINDOW = 512
COUNTS = {"train": 128, "validation": 32}
SHARD_NAME = re.compile(r"(train|validation)-(\d{5})-of-(\d{5})\.parquet")
CHUNK_BYTES = 1 << 20


def digest_file(path: Path) -> str:
    """Return a SHA-256 digest of a dataset, token, or script file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_shards(source: Path, split: str) -> list[Path]:
    """Find the complete, ordered set of pinned Parquet split shards."""
    matches: list[tuple[int, int, Path]] = []
    for path in source.glob(f"{split}-*.parquet"):
        match = SHARD_NAME.fullmatch(path.name)
        if match and match.group(1) == split:
            matches.append((int(match.group(2)), int(match.group(3)), path))
    matches.sort()
    if not matches or any(
        total != len(matches) or index != position
        for position, (index, total, _) in enumerate(matches)
    ):
        raise ValueError(f"incomplete or malformed {split} shard sequence")
    return [path for _, _, path in matches]


def selected_offsets(split: str, token_count: int) -> list[tuple[str, int]]:
    """Rank disjoint windows by the frozen SHA-256 byte ordering."""
    ranking = []
    for offset in range(0, token_count - WINDOW + 1, WINDOW):
        seed = f"qw38-task020-{split}-v1\n{REVISION}\n{offset}".encode()
        ranking.append((hashlib.sha256(seed).digest(), offset))
    ranking.sort(key=lambda item: (item[0], item[1]))
    if len(ranking) < COUNTS[split]:
        raise ValueError(f"{split} has too few complete windows")
    return [(digest.hex(), offset) for digest, offset in ranking[: COUNTS[split]]]


def text_chunks(path: Path, limit: int = CHUNK_BYTES) -> Iterator[str]:
    """Split at LF before space-letter, a tokenizer pre-token boundary."""
    parts: list[bytes] = []
    size = 0
    with path.open("rb") as source:
        for line in source:
            if (
                size >= limit
                and len(line) > 1
                and line[0] == 32
                and line[1] in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                and parts[-1].endswith(b"\n")
            ):
                yield b"".join(parts).decode("utf-8")
                parts = []
                size = 0
            parts.append(line)
            size += len(line)
            if size > 8 * CHUNK_BYTES:
                raise ValueError("no tokenizer-safe split in eight MiB of source text")
    if parts:
        yield b"".join(parts).decode("utf-8")


def tokenize_joined(tokenizer, joined_path: Path, token_path: Path) -> int:
    """Tokenize the joined source with bounded memory and write little-endian IDs."""
    count = 0
    with token_path.open("wb") as tokens_file:
        for chunk in text_chunks(joined_path):
            ids = tokenizer.encode(chunk, add_special_tokens=False).ids
            if any(token < 0 or token >= 248320 for token in ids):
                raise ValueError("tokenizer produced an out-of-vocabulary token ID")
            for start in range(0, len(ids), 1 << 18):
                part = ids[start : start + (1 << 18)]
                tokens_file.write(struct.pack(f"<{len(part)}I", *part))
            count += len(ids)
    return count


def materialize(source: Path, tokenizer_path: Path, output: Path) -> None:
    """Write source, row, token, and selected-window identities before fitting."""
    from pyarrow import parquet
    from tokenizers import Tokenizer

    if digest_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("frozen tokenizer.json identity differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output.mkdir(parents=True, exist_ok=True)
    script_sha = digest_file(Path(__file__))
    for split in COUNTS:
        shards = ordered_shards(source, split)
        joined_path = output / f"{split}.joined.utf8"
        row_digest_path = output / f"{split}.rows.sha256le"
        row_count = 0
        shard_records = []
        with (
            joined_path.open("wb") as joined,
            row_digest_path.open("wb") as row_digests,
        ):
            for shard in shards:
                rows_in_shard = 0
                reader = parquet.ParquetFile(shard)
                if "text" not in reader.schema.names:
                    raise ValueError(f"missing text column: {shard}")
                for batch in reader.iter_batches(batch_size=8192, columns=["text"]):
                    for value in batch.column(0).to_pylist():
                        if not isinstance(value, str):
                            raise TypeError(f"non-string text row: {shard}")
                        raw = value.encode("utf-8")
                        if row_count:
                            joined.write(b"\n")
                        joined.write(raw)
                        row_digests.write(hashlib.sha256(raw).digest())
                        row_count += 1
                        rows_in_shard += 1
                shard_records.append(
                    {
                        "path": str(shard),
                        "bytes": shard.stat().st_size,
                        "sha256": digest_file(shard),
                        "rows": rows_in_shard,
                    }
                )
        token_path = output / f"{split}.tokens.u32le"
        token_count = tokenize_joined(tokenizer, joined_path, token_path)
        windows = []
        with token_path.open("rb") as tokens_file:
            for rank, (selector_sha, offset) in enumerate(
                selected_offsets(split, token_count)
            ):
                path = output / f"{split}.window-{rank:03d}.u32le"
                tokens_file.seek(offset * 4)
                part = tokens_file.read(WINDOW * 4)
                if len(part) != WINDOW * 4:
                    raise ValueError("selected token window is truncated")
                path.write_bytes(part)
                windows.append(
                    {
                        "rank": rank,
                        "offset": offset,
                        "selector_sha256": selector_sha,
                        "tokens_file": path.name,
                        "tokens_sha256": digest_file(path),
                    }
                )
        manifest = {
            "schema": "qw38-task020-token-manifest-v1",
            "role": "calibration" if split == "train" else "development-screening",
            "dataset": "Salesforce/wikitext",
            "configuration": "wikitext-103-raw-v1",
            "revision": REVISION,
            "split": split,
            "shards": shard_records,
            "row_count": row_count,
            "row_digest_file": row_digest_path.name,
            "row_digest_sha256": digest_file(row_digest_path),
            "joined_text_file": joined_path.name,
            "joined_text_sha256": digest_file(joined_path),
            "tokenizer": {
                "implementation": "tokenizers",
                "version": importlib.metadata.version("tokenizers"),
                "tokenizer_json_sha256": TOKENIZER_SHA256,
                "add_special_tokens": False,
                "chat_template": False,
            },
            "token_count": token_count,
            "token_file": token_path.name,
            "token_sha256": digest_file(token_path),
            "sampler_script_sha256": script_sha,
            "window_length": WINDOW,
            "window_count": len(windows),
            "windows": windows,
        }
        (output / f"{split}.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "split": split,
                    "rows": row_count,
                    "tokens": token_count,
                    "windows": len(windows),
                }
            ),
            flush=True,
        )


def main() -> None:
    """Read command-line paths and create the frozen token manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
