#!/usr/bin/env python3
"""Prepare exact token windows for paired llama.cpp development scoring."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--role",
        choices=("development-screening", "calibration"),
        default="development-screening",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["role"] != args.role or args.limit < 1:
        raise ValueError("window role differs")
    rows = []
    for window in manifest["windows"][: args.limit]:
        path = args.manifest.parent / window["tokens_file"]
        data = path.read_bytes()
        if (
            len(data) != 512 * 4
            or hashlib.sha256(data).hexdigest() != window["tokens_sha256"]
        ):
            raise ValueError(f"window identity differs: {path}")
        tokens = struct.unpack("<512I", data)
        prompt = struct.pack("<384I", *tokens[:384])
        target = struct.pack("<128I", *tokens[384:])
        rows.append(
            (
                f"{manifest['split']}.window-{window['rank']:03d}",
                str(path),
                hashlib.sha256(prompt).hexdigest(),
                hashlib.sha256(target).hexdigest(),
            )
        )
    args.output.write_text("".join("\t".join(row) + "\n" for row in rows))
    print(f"cases={len(rows)} target_tokens={len(rows) * 128}")


if __name__ == "__main__":
    main()
