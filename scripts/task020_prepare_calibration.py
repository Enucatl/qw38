#!/usr/bin/env python3
"""Decode frozen training windows for llama.cpp activation importance collection."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["role"] != "calibration" or manifest["split"] != "train":
        raise ValueError("expected frozen calibration train windows")
    if (
        hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        != manifest["tokenizer"]["tokenizer_json_sha256"]
    ):
        raise ValueError("tokenizer identity differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    segments = []
    text = ""
    ids = []
    start_rank = 0
    for window in manifest["windows"]:
        path = args.manifest.parent / window["tokens_file"]
        data = path.read_bytes()
        if (
            len(data) != 512 * 4
            or hashlib.sha256(data).hexdigest() != window["tokens_sha256"]
        ):
            raise ValueError(f"window identity differs: {path}")
        window_ids = list(struct.unpack("<512I", data))
        decoded = tokenizer.decode(window_ids, skip_special_tokens=False)
        if tokenizer.encode(decoded, add_special_tokens=False).ids != window_ids:
            raise ValueError(f"window does not round-trip: {path}")
        candidate = text + decoded
        if (
            ids
            and tokenizer.encode(candidate, add_special_tokens=False).ids
            != ids + window_ids
        ):
            segments.append((start_rank, window["rank"], text, ids))
            text, ids, start_rank = "", [], window["rank"]
            candidate = decoded
        text = candidate
        ids += window_ids
    segments.append((start_rank, len(manifest["windows"]), text, ids))
    rows = []
    for number, (first, stop, segment_text, segment_ids) in enumerate(segments):
        if tokenizer.encode(segment_text, add_special_tokens=False).ids != segment_ids:
            raise ValueError(f"segment does not reproduce exact tokens: {number}")
        path = args.output.with_name(f"{args.output.name}.segment-{number:03d}.txt")
        path.write_text(segment_text, encoding="utf-8")
        rows.append(
            {
                "path": str(path),
                "first_rank": first,
                "stop_rank": stop,
                "chunks": stop - first,
                "tokens": len(segment_ids),
                "text_sha256": hashlib.sha256(segment_text.encode()).hexdigest(),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "schema": "qw38-task020-exact-calibration-segments-v1",
                "source_manifest": str(args.manifest),
                "segments": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"windows={len(manifest['windows'])} segments={len(segments)} tokens={sum(row['tokens'] for row in rows)}"
    )


if __name__ == "__main__":
    main()
