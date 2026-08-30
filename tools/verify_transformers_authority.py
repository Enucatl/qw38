"""Fail-closed identity checks for the official Transformers authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int | None, expected_hash: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing authority file: {path.name}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"authority file size differs: {path.name}")
    if sha256(path) != expected_hash:
        raise ValueError(f"authority file hash differs: {path.name}")


def verify(contract_path: Path, checkpoint: Path, source: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != contract["transformers"]["revision"]:
        raise ValueError("Transformers source revision differs")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("Transformers authority checkout is dirty")

    shards = contract["checkpoint"]["shards"]
    expected_names = {shard["file"] for shard in shards}
    actual_names = {path.name for path in checkpoint.glob("model-*.safetensors")}
    if actual_names != expected_names:
        raise ValueError("checkpoint shard set differs")
    for shard in shards:
        verify_file(checkpoint / shard["file"], int(shard["bytes"]), shard["sha256"])
    for name, digest in contract["checkpoint"]["support_files"].items():
        verify_file(checkpoint / name, None, digest)

    return {
        "schema_version": contract["version"],
        "checkpoint_revision": contract["checkpoint"]["revision"],
        "transformers_revision": revision,
        "shard_count": len(shards),
        "shard_bytes": sum(int(shard["bytes"]) for shard in shards),
        "identity": "verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract, args.checkpoint, args.source)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
