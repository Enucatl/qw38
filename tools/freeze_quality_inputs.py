"""Create immutable QLT-001 inputs from a local WikiText parquet file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DATASET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
MAX_CONTEXT = 131072
TASKS = (
    ("task_arithmetic", "What is 17 + 25?\nA. 41\nB. 42\nC. 43\nD. 44"),
    (
        "task_python_len",
        'What does this Python expression return: len("quartz")?\nA. 6\nB. 5\nC. 7\nD. 8',
    ),
    (
        "task_inference",
        "All quartz watches are timepieces. This object is a quartz watch. Which conclusion follows?\nA. It is not a timepiece.\nB. It might be a calendar.\nC. It is a timepiece.\nD. No conclusion follows.",
    ),
    ("task_minutes", "How many minutes are in 3 hours?\nA. 60\nB. 120\nC. 240\nD. 180"),
    (
        "task_sort",
        "Which list is in ascending numeric order?\nA. 9, 5, 2\nB. 2, 5, 9\nC. 5, 2, 9\nD. 2, 9, 5",
    ),
    (
        "task_json",
        'Which option is a valid JSON object?\nA. {"x":1}\nB. {\'x\':1}\nC. {x:1}\nD. ["x":1]',
    ),
    (
        "task_reading",
        "Mira placed an amber key in the blue box. She placed a silver key in the red box. Which key is in the blue box?\nA. silver\nB. red\nC. amber\nD. blue",
    ),
    (
        "task_sequence",
        "What number comes next: 2, 4, 8, 16, ?\nA. 18\nB. 20\nC. 24\nD. 32",
    ),
)
ANSWERS = dict(zip((name for name, _ in TASKS), "BACDBACD"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def build_record(dataset: Path, tokenizer_path: Path) -> dict[str, Any]:
    if digest(dataset) != DATASET_SHA256 or digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("dataset or tokenizer hash does not match pinned identity")
    try:
        import pyarrow.parquet as parquet
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(f"freezer dependencies unavailable: {error}") from error
    values = parquet.read_table(dataset).column("text")
    text = "\n".join(str(value.as_py()) for value in values if value.as_py())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stream = _encode(tokenizer, text)
    if len(stream) < 24576:
        raise ValueError("WikiText stream is too short for frozen offsets")
    cases: dict[str, dict[str, Any]] = {
        "wikitext_nll": {"context": stream[:1], "continuation": stream[1:1025]}
    }
    for offset in (2048, 4096, 6144, 8192):
        cases[f"continuation_{offset}"] = {
            "context": stream[offset - 128 : offset],
            "continuation": stream[offset : offset + 16],
        }
    target_start = 20480
    target = stream[target_start : target_start + 128]
    cases["recurrence_short"] = {
        "context": stream[target_start - 256 : target_start],
        "continuation": target,
    }
    cases["recurrence_long"] = {
        "context": stream[target_start - 4096 : target_start],
        "continuation": target,
    }
    prefix = "Memorize this access code: 7391.\n"
    suffix = "\nQuestion: What access code were you told to memorize? Reply with the four digits only:\n"
    distractor = stream[32768:]
    if "7391" in tokenizer.decode(distractor, skip_special_tokens=False):
        raise ValueError("distractor contains retrieval needle")
    prefix_ids, suffix_ids, answer_ids = (
        _encode(tokenizer, prefix),
        _encode(tokenizer, suffix),
        _encode(tokenizer, "7391"),
    )
    available = MAX_CONTEXT - len(prefix_ids) - len(suffix_ids) - len(answer_ids)
    repeated = (distractor * ((available // max(1, len(distractor))) + 1))[:available]
    retrieval = prefix_ids + repeated + suffix_ids
    if len(retrieval) + len(answer_ids) != MAX_CONTEXT:
        raise ValueError("retrieval construction did not reach 131072 tokens")
    cases["retrieval_128k"] = {
        "context": retrieval,
        "continuation": answer_ids,
        "prefix": prefix,
        "suffix": suffix,
        "needle": "7391",
        "needle_position": 0,
    }
    for name, question in TASKS:
        rendered = question + "\nReply with exactly one letter: A, B, C, or D."
        answer = ANSWERS[name]
        ids, answer_token = _encode(tokenizer, rendered), _encode(tokenizer, answer)
        if len(answer_token) != 1:
            raise ValueError(f"answer {answer!r} is not one token")
        cases[name] = {
            "user": question,
            "rendered": rendered,
            "expected": answer,
            "context": ids,
            "continuation": answer_token,
        }
    return {
        "schema": "qw38.quality-inputs",
        "version": 1,
        "dataset_sha256": DATASET_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "stream_token_count": len(stream),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = build_record(args.dataset, args.tokenizer)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    temporary = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
