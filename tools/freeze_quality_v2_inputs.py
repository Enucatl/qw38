"""Freeze chat-templated v2 quality inputs without rewriting QLT-001."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.opt058_quality_baseline import (
    TOKEN_271,
    build_v2_inputs,
    bundle_cases,
    decode_token_271,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-inputs",
        type=Path,
        default=ROOT / "fixtures/quality_inputs.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models/tokenizer/tokenizer.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pins/production_quality_v2_inputs.json",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "pins/production_quality_v2_nll.bundle",
    )
    parser.add_argument(
        "--functional-bundle",
        type=Path,
        default=ROOT / "pins/production_quality_v2_functional.bundle",
    )
    parser.add_argument(
        "--original-bundle",
        type=Path,
        default=ROOT / "pins/production_quality_original_functional.bundle",
    )
    args = parser.parse_args()
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        parser.error(f"tokenizer package unavailable: {error}")
    quality = json.loads(args.quality_inputs.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    record = build_v2_inputs(quality, tokenizer)
    record["token_271_decoded"] = decode_token_271(tokenizer)
    record["token_271"] = TOKEN_271
    write_json(args.output, record)
    from tools.opt058_quality_baseline import NLL_CASES, TASK_NAMES

    bundle_cases(record, args.bundle, NLL_CASES)
    bundle_cases(record, args.functional_bundle, TASK_NAMES)
    original = {"cases": {}}
    for name in TASK_NAMES:
        source = quality["cases"][name]
        original["cases"][name] = {
            "context": source["context"],
            "continuation": source["continuation"],
        }
    bundle_cases(original, args.original_bundle, TASK_NAMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
