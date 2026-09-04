"""Align and report comparable Quartz, Transformers, and llama.cpp taps."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

from tools.qw38_trace import compare_values

ROOT = Path(__file__).resolve().parents[1]
LAYERS = (0, 3, 7, 62, 63)
GDN_LAYERS = frozenset((0, 62))


def fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[a-z][a-z0-9_]*", key) is None:
            continue
        if key in result:
            raise ValueError(f"duplicate authority field: {key}")
        result[key] = value
    return result


def read_values(path: Path, offset: int, count: int) -> tuple[float, ...]:
    with path.open("rb") as source:
        source.seek(offset)
        raw = source.read(count * 4)
    if len(raw) != count * 4:
        raise ValueError(f"short tensor read from {path}")
    return struct.unpack(f"<{count}f", raw)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quartz_records(
    metadata: dict[str, str],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index in range(int(metadata["tensor_count"])):
        prefix = f"tensor_{index}_"
        key = (
            int(metadata[f"{prefix}position"]),
            int(metadata[f"{prefix}layer"]),
            metadata[f"{prefix}name"],
        )
        if key in records:
            raise ValueError(f"duplicate Quartz tensor: {key}")
        records[key] = {
            "offset": int(metadata[f"{prefix}offset"]),
            "count": int(metadata[f"{prefix}count"]),
            "shape": [int(value) for value in metadata[f"{prefix}shape"].split(",")],
        }
    return records


def llama_records(
    metadata: dict[str, str],
) -> dict[tuple[int, str, tuple[int, ...]], dict[str, Any]]:
    records: dict[tuple[int, str, tuple[int, ...]], dict[str, Any]] = {}
    for index in range(int(metadata["trace_tensor_count"])):
        prefix = f"trace_{index}_"
        shape = tuple(int(value) for value in metadata[f"{prefix}shape"].split(","))
        key = (
            int(metadata[f"{prefix}position"]),
            metadata[f"{prefix}name"],
            shape,
        )
        if key in records:
            raise ValueError(f"duplicate llama.cpp tensor: {key}")
        records[key] = {
            "offset": int(metadata[f"{prefix}offset"]),
            "count": int(metadata[f"{prefix}count"]),
            "shape": list(shape),
        }
    return records


def official_records(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {record["name"]: record for record in run["taps"]["records"]}
    if len(records) != len(run["taps"]["records"]):
        raise ValueError("duplicate official tensor name")
    return records


def metric(expected: tuple[float, ...], actual: tuple[float, ...]) -> dict[str, Any]:
    return dataclasses.asdict(
        compare_values(
            expected,
            actual,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            top_k_logits=10 if len(expected) == 248320 else None,
        )
    )


def gdn_permute(
    values: tuple[float, ...], head_width: int, *, grouped_to_tiled: bool
) -> tuple[float, ...]:
    expected = 16 * 3 * head_width
    if len(values) != expected:
        raise ValueError(f"GDN permutation expected {expected} values")
    output = [0.0] * expected
    for key in range(16):
        for replica in range(3):
            grouped = (key * 3 + replica) * head_width
            tiled = (replica * 16 + key) * head_width
            source = grouped if grouped_to_tiled else tiled
            destination = tiled if grouped_to_tiled else grouped
            output[destination : destination + head_width] = values[
                source : source + head_width
            ]
    return tuple(output)


def normalize_official(boundary: str, values: tuple[float, ...]) -> tuple[float, ...]:
    if boundary in ("gdn.packed_qkv", "gdn.convolution"):
        return values[:4096] + gdn_permute(values[4096:], 128, grouped_to_tiled=True)
    if boundary == "gdn.convolution_state":
        return values[: 4096 * 4] + gdn_permute(
            values[4096 * 4 :], 128 * 4, grouped_to_tiled=True
        )
    return values


def normalize_llama(boundary: str, values: tuple[float, ...]) -> tuple[float, ...]:
    widths = {
        "gdn.value": 128,
        "gdn.decay": 1,
        "gdn.beta": 1,
        "gdn.recurrent_output": 128,
        "gdn.gated_output": 128,
    }
    width = widths.get(boundary)
    if width is None:
        return values
    return gdn_permute(values, width, grouped_to_tiled=False)


def mappings(layer: int) -> list[tuple[str, str, str, tuple[int, ...] | None]]:
    prefix = f"layer.{layer}"
    common = [
        ("input_norm", f"{prefix}.input_norm.output", "input_norm", (5120, 1, 1, 1)),
        (
            "mixer_residual",
            f"{prefix}.mixer_residual.output",
            "mixer_residual",
            (5120, 1, 1, 1),
        ),
        (
            "ffn.input_norm",
            f"{prefix}.post_mixer_norm.output",
            "ffn.input_norm",
            (5120, 1, 1, 1),
        ),
        ("ffn.gate", f"{prefix}.ffn.gate_projection", "ffn.gate", None),
        ("ffn.up", f"{prefix}.ffn.up_projection", "ffn.up", None),
        ("ffn.activated", f"{prefix}.ffn.activated", "ffn.activated", None),
        ("ffn.correction", f"{prefix}.ffn.output", "ffn.correction", (5120, 1, 1, 1)),
        (
            "layer_residual",
            f"{prefix}.residual.output",
            "layer_residual",
            (5120, 1, 1, 1),
        ),
    ]
    if layer in GDN_LAYERS:
        mixer = [
            (
                "gdn.packed_qkv",
                f"{prefix}.gdn.qkv_projection",
                "gdn.packed_qkv",
                (10240, 1, 1, 1),
            ),
            (
                "gdn.convolution",
                f"{prefix}.gdn.convolution_current",
                "gdn.convolution",
                (10240, 1, 1, 1),
            ),
            ("gdn.query", f"{prefix}.gdn.query_unique", "gdn.query", (128, 16, 1, 1)),
            ("gdn.key", f"{prefix}.gdn.key_unique", "gdn.key", (128, 16, 1, 1)),
            ("gdn.value", f"{prefix}.gdn.value", "gdn.value", (128, 48, 1, 1)),
            ("gdn.decay", f"{prefix}.gdn.g", "gdn.decay", (48, 1, 1, 1)),
            ("gdn.beta", f"{prefix}.gdn.beta", "gdn.beta", (1, 48, 1, 1)),
            (
                "gdn.recurrent_output",
                f"{prefix}.gdn.core_output",
                "gdn.recurrent_output",
                (128, 48, 1, 1),
            ),
            (
                "gdn.recurrent_state",
                f"{prefix}.gdn.recurrent_state",
                "gdn.recurrent_state",
                None,
            ),
            (
                "gdn.convolution_state",
                f"{prefix}.gdn.convolution_state",
                "gdn.convolution_state",
                None,
            ),
            (
                "gdn.gated_output",
                f"{prefix}.gdn.gated_norm",
                "gdn.gated_output",
                (6144, 1, 1, 1),
            ),
            ("gdn.output", f"{prefix}.gdn.output", "gdn.output", (5120, 1, 1, 1)),
        ]
    else:
        mixer = [
            (
                "attention.packed_query_gate",
                f"{prefix}.attention.query_gate_projection",
                "attention.packed_query_gate",
                (12288, 1, 1, 1),
            ),
            (
                "attention.query",
                f"{prefix}.attention.query_gate_projection",
                "attention.query",
                (256, 24, 1, 1),
            ),
            (
                "attention.key",
                f"{prefix}.attention.key_projection",
                "attention.key",
                (1024, 1, 1, 1),
            ),
            (
                "attention.rope_query",
                f"{prefix}.attention.query_after_rope",
                "attention.rope_query",
                (256, 24, 1, 1),
            ),
            (
                "attention.rope_key",
                f"{prefix}.attention.key_after_rope",
                "attention.rope_key",
                (256, 4, 1, 1),
            ),
            (
                "attention.value",
                f"{prefix}.attention.value_projection",
                "attention.value",
                (256, 4, 1, 1),
            ),
            (
                "attention.kv_key_row",
                f"{prefix}.attention.key_row",
                "attention.kv_key_row",
                (256, 4, 1, 1),
            ),
            (
                "attention.kv_value_row",
                f"{prefix}.attention.value_row",
                "attention.kv_value_row",
                (256, 4, 1, 1),
            ),
            (
                "attention.context",
                f"{prefix}.attention.gated_context",
                "attention.context",
                (6144, 1, 1, 1),
            ),
            (
                "attention.output",
                f"{prefix}.attention.output",
                "attention.output",
                (5120, 1, 1, 1),
            ),
        ]
    return mixer + common


LLAMA_NAMES = {
    "input_norm": "attn_norm",
    "mixer_residual": "attn_residual",
    "ffn.input_norm": "attn_post_norm",
    "ffn.correction": "ffn_out",
    "layer_residual": "l_out",
    "gdn.packed_qkv": "linear_attn_qkv_mixed",
    "gdn.convolution": "conv_output_silu",
    "gdn.query": "q_conv",
    "gdn.key": "k_conv",
    "gdn.value": "v_conv_predelta",
    "gdn.decay": "gate",
    "gdn.beta": "beta_sigmoid",
    "gdn.recurrent_output": "attn_output",
    "gdn.gated_output": "final_output",
    "gdn.output": "linear_attn_out",
    "attention.packed_query_gate": "Qcur_full",
    "attention.query": "Qcur_reshaped",
    "attention.key": "Kcur",
    "attention.rope_query": "Qcur",
    "attention.rope_key": "Kcur",
    "attention.value": "Vcur",
    "attention.kv_key_row": "Kcur",
    "attention.kv_value_row": "Vcur",
    "attention.context": "attn_gated",
    "attention.output": "attn_output",
}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    quartz_meta = fields(args.quartz_fields)
    llama_meta = fields(args.llama_fields)
    official_run = json.loads(args.official_run.read_text(encoding="utf-8"))
    quartz = quartz_records(quartz_meta)
    llama = llama_records(llama_meta)
    official = official_records(official_run)
    rows: list[dict[str, Any]] = []

    def add(
        position: int,
        layer: int,
        boundary: str,
        official_name: str,
        quartz_name: str,
        llama_shape: tuple[int, ...] | None,
        llama_name: str | None,
    ) -> None:
        official_record = official[f"position.{position}.{official_name}"]
        quartz_record = quartz[(position, layer, quartz_name)]
        expected = normalize_official(
            boundary,
            read_values(
                args.official_blob,
                official_record["offset"],
                official_record["bytes"] // 4,
            ),
        )
        if boundary == "attention.query":
            expected = tuple(
                value
                for head in range(24)
                for value in expected[head * 512 : head * 512 + 256]
            )
        actual = read_values(
            args.quartz_blob, quartz_record["offset"], quartz_record["count"]
        )
        if len(expected) != len(actual):
            raise ValueError(
                f"length mismatch at position {position}, layer {layer}, "
                f"boundary {boundary}: official {len(expected)}, Quartz {len(actual)}"
            )
        row: dict[str, Any] = {
            "position": position,
            "layer": None if layer == 64 else layer,
            "boundary": boundary,
            "count": len(expected),
            "official_vs_quartz": metric(expected, actual),
        }
        if llama_name is not None and llama_shape is not None:
            full_name = llama_name if layer == 64 else f"{llama_name}-{layer}"
            llama_record = llama[(position, full_name, llama_shape)]
            llama_values = normalize_llama(
                boundary,
                read_values(
                    args.llama_blob,
                    llama_record["offset"],
                    llama_record["count"],
                ),
            )
            row["llama_vs_quartz"] = metric(llama_values, actual)
        rows.append(row)

    for position in range(2):
        add(
            position,
            64,
            "embedding",
            "embedding.output",
            "embedding",
            (5120, 1, 1, 1),
            "model.input_embed",
        )
        for layer in LAYERS:
            for boundary, official_name, quartz_name, llama_shape in mappings(layer):
                add(
                    position,
                    layer,
                    boundary,
                    official_name,
                    quartz_name,
                    llama_shape,
                    LLAMA_NAMES.get(boundary),
                )
        add(
            position,
            64,
            "final_norm",
            "final_norm.output",
            "final_norm",
            (5120, 1, 1, 1),
            "result_norm",
        )
        add(
            position,
            64,
            "logits",
            "logits",
            "logits",
            (248320, 1, 1, 1),
            "result_output",
        )
    return {
        "schema_version": 1,
        "admission_status": "reporting_only_tolerances_not_frozen",
        "input_tokens": [42, 3649],
        "identities": {
            "model_gguf_sha256": "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34",
            "official_checkpoint_revision": official_run["identity"][
                "checkpoint_revision"
            ],
            "transformers_revision": official_run["identity"]["transformers_revision"],
            "llama_cpp_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
            "blobs": {
                "official_taps_sha256": official_run["taps"]["sha256"],
                "quartz_sha256": sha256(args.quartz_blob),
                "llama_cpp_sha256": sha256(args.llama_blob),
            },
        },
        "comparisons": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--official-blob", type=Path, required=True)
    parser.add_argument("--quartz-fields", type=Path, required=True)
    parser.add_argument("--quartz-blob", type=Path, required=True)
    parser.add_argument("--llama-fields", type=Path, required=True)
    parser.add_argument("--llama-blob", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare(args), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
