"""Compare complete Quartz and pinned llama.cpp scalar logit rows."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import struct
from pathlib import Path

from tools.qw38_trace import compare_values

_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]*")


def parse_fields(path: Path) -> dict[str, str]:
    """Read the deliberately small key=value authority protocol."""

    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if _FIELD_NAME.fullmatch(key) is None:
            continue
        if key in fields:
            raise ValueError(f"duplicate authority field: {key!r}")
        fields[key] = value
    return fields


def load_rows(path: Path, rows: int, width: int) -> tuple[tuple[float, ...], ...]:
    """Load exact little-endian FP32 rows after checking their byte count."""

    raw = path.read_bytes()
    expected_bytes = rows * width * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path} has {len(raw)} bytes; expected exactly {expected_bytes}"
        )
    values = struct.unpack(f"<{rows * width}f", raw)
    return tuple(values[row * width : (row + 1) * width] for row in range(rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    native_logits: Path,
    llama_logits: Path,
    native_stdout: Path,
    llama_stdout: Path,
    contract_path: Path,
    template_bytes: Path | None = None,
    native_template_ids: Path | None = None,
    llama_template_stdout: Path | None = None,
) -> dict[str, object]:
    """Validate identities and return reporting-only, zero-tolerance metrics."""

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_tokens = contract["diagnostic"]["input_tokens"]
    native = parse_fields(native_stdout)
    llama = parse_fields(llama_stdout)
    token_count = len(expected_tokens)
    vocabulary_size = int(native["vocabulary_size"])
    if int(llama["vocabulary_size"]) != vocabulary_size:
        raise ValueError("authority vocabulary sizes differ")
    if (
        int(native["token_count"]) != token_count
        or int(llama["token_count"]) != token_count
    ):
        raise ValueError("authority token counts differ from the contract")
    for index, expected in enumerate(expected_tokens):
        if int(native[f"token_{index}"]) != expected:
            raise ValueError("Quartz token sequence differs from the contract")
        if int(llama[f"token_{index}"]) != expected:
            raise ValueError("llama.cpp token sequence differs from the contract")

    native_rows = load_rows(native_logits, token_count, vocabulary_size)
    llama_rows = load_rows(llama_logits, token_count, vocabulary_size)
    comparisons: list[dict[str, object]] = []
    for index, (llama_row, native_row) in enumerate(
        zip(llama_rows, native_rows, strict=True)
    ):
        metrics = compare_values(
            llama_row,
            native_row,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            top_k_logits=10,
        )
        comparisons.append(
            {
                "position": index,
                "input_token": expected_tokens[index],
                "llama_greedy_token": int(llama[f"greedy_{index}"]),
                "quartz_greedy_token": int(native[f"greedy_{index}"]),
                "metrics": dataclasses.asdict(metrics),
            }
        )
    document: dict[str, object] = {
        "schema_version": 1,
        "authority": "pinned llama.cpp same-GGUF independent oracle",
        "comparison_direction": {
            "expected": "llama.cpp CUDA",
            "actual": "Quartz scalar CPU",
        },
        "admission_status": "reporting_only_tolerances_not_frozen",
        "model": contract["model"],
        "llama_source": contract["source"],
        "input_tokens": expected_tokens,
        "vocabulary_size": vocabulary_size,
        "logits": {
            "dtype": "float32",
            "endianness": "little",
            "row_count": token_count,
            "row_width": vocabulary_size,
            "native_sha256": sha256(native_logits),
            "llama_sha256": sha256(llama_logits),
        },
        "llama_version": llama["llama_version"],
        "comparisons": comparisons,
        "proof_limit": (
            "independent same-GGUF logits and greedy continuation only; "
            "Transformers semantic authority and frozen tolerances remain ORA-003/ORA-004"
        ),
    }
    template_inputs = (template_bytes, native_template_ids, llama_template_stdout)
    if any(path is not None for path in template_inputs):
        if not all(path is not None for path in template_inputs):
            raise ValueError("all template comparison inputs are required together")
        assert template_bytes is not None
        assert native_template_ids is not None
        assert llama_template_stdout is not None
        native_ids = tuple(
            int(value)
            for value in native_template_ids.read_text(encoding="utf-8")
            .strip()
            .split(",")
        )
        template_fields = parse_fields(llama_template_stdout)
        llama_ids = tuple(
            int(value) for value in template_fields["token_ids"].split(",")
        )
        raw_template = template_bytes.read_bytes()
        if int(template_fields["template_byte_count"]) != len(raw_template):
            raise ValueError("llama.cpp template byte count differs")
        if int(template_fields["token_count"]) != len(llama_ids):
            raise ValueError("llama.cpp template token count differs")
        if native_ids != llama_ids:
            raise ValueError("Quartz and llama.cpp template token IDs differ")
        document["template_identity"] = {
            "case": contract["diagnostic"]["template_case"],
            "byte_count": len(raw_template),
            "sha256": hashlib.sha256(raw_template).hexdigest(),
            "token_ids": list(native_ids),
            "exact_equal": True,
        }
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--native-stdout", type=Path, required=True)
    parser.add_argument("--llama-stdout", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--template-bytes", type=Path, required=True)
    parser.add_argument("--native-template-ids", type=Path, required=True)
    parser.add_argument("--llama-template-stdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = compare(
        args.native_logits,
        args.llama_logits,
        args.native_stdout,
        args.llama_stdout,
        args.contract,
        args.template_bytes,
        args.native_template_ids,
        args.llama_template_stdout,
    )
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
