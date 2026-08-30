"""Build the small checked-in manifest for a real Transformers authority run."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from tools.compare_llama_authority import load_rows, sha256
from tools.qw38_trace import compare_values


def comparisons(
    official_logits: Path,
    quartz_logits: Path,
    llama_logits: Path,
    rows: int,
    width: int,
) -> list[dict[str, Any]]:
    official = load_rows(official_logits, rows, width)
    quartz = load_rows(quartz_logits, rows, width)
    llama = load_rows(llama_logits, rows, width)
    result: list[dict[str, Any]] = []
    for position in range(rows):
        result.append(
            {
                "position": position,
                "official_greedy_token": max(
                    range(width), key=official[position].__getitem__
                ),
                "quartz": dataclasses.asdict(
                    compare_values(
                        official[position],
                        quartz[position],
                        absolute_tolerance=0.0,
                        relative_tolerance=0.0,
                        top_k_logits=10,
                    )
                ),
                "llama_cpp": dataclasses.asdict(
                    compare_values(
                        official[position],
                        llama[position],
                        absolute_tolerance=0.0,
                        relative_tolerance=0.0,
                        top_k_logits=10,
                    )
                ),
            }
        )
    return result


def freeze(
    run_path: Path,
    contract_path: Path,
    official_logits: Path,
    quartz_logits: Path,
    llama_logits: Path,
) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tokens = contract["execution"]["input_tokens"]
    width = int(run["vocabulary_size"])
    return {
        "schema_version": 1,
        "authority": "pinned official-checkpoint Transformers eager execution",
        "admission_status": "reporting_only_tolerances_not_frozen",
        "identity": run["identity"],
        "environment": {
            "python": contract["environment"]["python"],
            "torch": run["torch"],
            "transformers_revision": run["identity"]["transformers_revision"],
            "attention": run["transformers_attention"],
        },
        "checkpoint": {
            "repository": contract["checkpoint"]["repository"],
            "revision": run["identity"]["checkpoint_revision"],
            "shard_count": run["identity"]["shard_count"],
            "shard_bytes": run["identity"]["shard_bytes"],
            "tensor_count": contract["checkpoint"]["tensor_count"],
            "tensor_bytes": contract["checkpoint"]["tensor_bytes"],
            "loaded_text_wrapper_tensor_count": 1184,
            "ignored_mtp_tensor_count": 15,
        },
        "execution": {
            "input_tokens": tokens,
            "greedy_tokens": run["greedy_tokens"],
            "device_map": run["device_map"],
            "load_seconds": run["load_seconds"],
            "execute_seconds": run["execute_seconds"],
            "cuda_peak_allocated_bytes": run["cuda_peak_allocated_bytes"],
            "cuda_peak_reserved_bytes": run["cuda_peak_reserved_bytes"],
            "maximum_rss_kib": run["maximum_rss_kib"],
        },
        "logits": {
            "dtype": "float32",
            "endianness": "little",
            "row_count": len(tokens),
            "row_width": width,
            "bytes": run["logits_bytes"],
            "official_sha256": sha256(official_logits),
            "quartz_sha256": sha256(quartz_logits),
            "llama_cpp_sha256": sha256(llama_logits),
        },
        "taps": run["taps"],
        "comparisons": comparisons(
            official_logits,
            quartz_logits,
            llama_logits,
            len(tokens),
            width,
        ),
        "proof_limit": (
            "official-checkpoint eager taps and logits are captured; cross-artifact "
            "differences are reporting-only until ORA-004 freezes per-tap tolerances"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--official-logits", type=Path, required=True)
    parser.add_argument("--quartz-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = freeze(
        args.run,
        args.contract,
        args.official_logits,
        args.quartz_logits,
        args.llama_logits,
    )
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
