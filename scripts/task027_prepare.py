# /// script
# requires-python = "==3.12.*"
# dependencies = ["transformers==5.17.0", "numpy==2.5.3"]
# ///
"""Freeze PERF-01 inputs and validate the comparator's public tokenizer API."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from task018_validate_fixtures import read_u32le, sha256_file

FIXTURES = Path(".cache/evaluation/qw38-language-v2")
SOURCE = Path(".cache/authorities/qwen3.8-27b-transformers")


def prepare(output: Path) -> None:
    """Materialize the unformatted case_000 stream with metadata identities."""
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    p100 = FIXTURES / "source/p100.jsonl"
    if sha256_file(p100) != manifest["sources"]["p100"]["sha256"]:
        raise ValueError("P100 source identity changed")
    for name, record in manifest["tokenizer"]["files"].items():
        if sha256_file(SOURCE / name) != record["sha256"]:
            raise ValueError(f"source metadata identity changed: {name}")
    prompt = next(
        json.loads(line)["prompt"]
        for line in p100.read_text().splitlines()
        if json.loads(line)["id"] == "case_000"
    )
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or tokenizer.decode(ids, clean_up_tokenization_spaces=False) != prompt:
        raise ValueError("source prompt roundtrip failed")
    (output / "prompt.txt").write_text(prompt)
    paths = {}
    for depth in (256, 512, 4096, 32768):
        stream = [ids[i % len(ids)] for i in range(depth + 128)]
        path = output / f"tokens-{depth}.u32le"
        path.write_bytes(struct.pack(f"<{len(stream)}I", *stream))
        paths[str(depth)] = {"path": str(path), "sha256": sha256_file(path)}
    record = {
        "status": "AWAITING_LLAMA_TOKENIZER_VALIDATION",
        "prompt": prompt,
        "source_ids": ids,
        "source_metadata": manifest["tokenizer"],
        "inputs": paths,
        "policy_sha256": sha256_file(Path("docs/architecture/evaluation-policy-v0.md")),
        "core_policy_sha256": sha256_file(
            Path("docs/architecture/evaluation-policy-core-54.md")
        ),
    }
    (output / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")


def validate(output: Path) -> None:
    """Check vocabulary strings, consumed bytes, special IDs and real encoding."""
    from transformers import AutoTokenizer

    record = json.loads((output / "inputs.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    ids = record["source_ids"]
    if read_u32le(output / "llama-vocab.u32le") != ids:
        raise ValueError("llama/source performance tokenization mismatch")
    vocab = tokenizer.get_vocab()
    mapping = {}
    for line in (output / "llama-vocab.tsv").read_text().splitlines():
        index, token, piece, attributes = line.split("\t")
        mapping[int(index)] = (
            bytes.fromhex(token).decode(),
            bytes.fromhex(piece),
            int(attributes),
        )
    if len(mapping) != 248320 or max(vocab.values()) >= len(mapping):
        raise ValueError("wrong vocabulary extent")
    for text, index in vocab.items():
        if mapping[index][0] != text:
            raise ValueError(f"vocabulary ID/text mismatch: {index}")
    # HF omits the model's unused padding IDs. Validate each explicitly.
    padding = sorted(set(mapping) - set(vocab.values()))
    if any(mapping[index][2] != 2 for index in padding):
        raise ValueError("GGUF extra vocabulary IDs are not UNUSED padding")
    for index in set(ids):
        decoded = tokenizer.decode([index], clean_up_tokenization_spaces=False).encode()
        if mapping[index][1] != decoded:
            raise ValueError(f"consumed token byte mismatch: {index}")
    metadata = {
        key: bytes.fromhex(value).decode()
        for key, value in (
            line.split("\t", 1)
            for line in (output / "llama-vocab.metadata.tsv").read_text().splitlines()
        )
    }
    expected = {
        "general.architecture": "qwen35",
        "qwen35.block_count": "64",
        "qwen35.embedding_length": "5120",
        "qwen35.feed_forward_length": "17408",
        "tokenizer.ggml.eos_token_id": str(tokenizer.eos_token_id),
        "tokenizer.ggml.padding_token_id": str(tokenizer.pad_token_id),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"GGUF metadata mismatch: {key}: {metadata.get(key)!r} != {value!r}"
            )
    if (
        tokenizer.bos_token_id is None
        and metadata.get("tokenizer.ggml.add_bos_token", "false") != "false"
    ):
        raise ValueError("unexpected GGUF automatic BOS")
    special_ids = sorted(tokenizer.all_special_ids)
    for index in special_ids:
        if mapping[index][1] != tokenizer.convert_ids_to_tokens(index).encode():
            raise ValueError(f"special-token byte mismatch: {index}")
    for item in record["inputs"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("frozen input changed")
    record.update(
        status="VALID",
        vocabulary_entries_checked=len(vocab),
        unused_padding_ids=padding,
        consumed_ids_checked=sorted(set(ids)),
        special_ids_checked=special_ids,
        gguf_metadata=metadata,
        gguf_size=Path("models/Qwen3.8-27B-Q4_K_M.gguf").stat().st_size,
        provenance={
            key: value
            for key, value in metadata.items()
            if any(tag in key for tag in ("source", "base_model", "name", "url"))
        },
        public_api_vocabulary_sha256=sha256_file(output / "llama-vocab.tsv"),
    )
    (output / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {"status": "VALID", "source_ids": ids, "vocabulary_entries": len(vocab)}
        )
    )


def main() -> None:
    """Prepare inputs or validate the independent public-API tokenizer dump."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "validate"))
    parser.add_argument("--output", type=Path, default=Path(".cache/task027"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else validate)(args.output)


if __name__ == "__main__":
    main()
