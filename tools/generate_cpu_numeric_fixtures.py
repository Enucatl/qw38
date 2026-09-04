#!/usr/bin/env python3
"""Generate 2B numeric fixtures. Run on the Darwin host with the pinned GGUF."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "build" / "qw38-eval"
MODEL = ROOT / "models" / "Qwen3.5-2B-Q4_K_M.gguf"
INVENTORY = ROOT / "pins" / "cpu_tensor_inventory.json"
FIXTURES = ROOT / "fixtures"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def parse_output(stdout: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)


def run_eval(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        [str(EVAL), *args],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )


def write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")
    print("wrote", path)


def generate_q5() -> None:
    q4_header = struct.pack("<e", 0.5) + struct.pack("<e", 0.25)
    q4_scales = bytes([0x00, 0x3F, 0xC0, 0xFF, 0x15, 0x2A, 0x95, 0xEA, 0xF0, 0x0F, 0xA5, 0x5A])
    q4_quants = bytes((index * 29 + 7) & 255 for index in range(128))
    high = bytes((index * 13 + 5) & 255 for index in range(32))
    pattern = q4_header + q4_scales + high + q4_quants
    zero = bytes(176)
    inventory = json.loads(INVENTORY.read_text())
    qkv = next(t for t in inventory["tensors"] if t["name"] == "blk.0.attn_qkv.weight")
    gguf = MODEL.read_bytes()
    artifact = gguf[qkv["absolute_offset"] : qkv["absolute_offset"] + 176]
    cases = []
    for name, block in (
        ("q5_packed_boundaries", pattern),
        ("q5_all_zero", zero),
        ("q5_blk0_attn_qkv_row0_block0", artifact),
    ):
        result = run_eval("--check-quant", "q5_k", block.hex())
        if result.returncode != 0:
            raise SystemExit(f"q5 {name}: {result.stderr}")
        actual = parse_output(result.stdout)
        cases.append(
            {
                "name": name,
                "kind": "q5_k",
                "block_hex": block.hex(),
                "decoded_f32_le_hex": actual["decoded_f32_le_hex"],
                "dot_f32_le_hex": actual["dot_f32_le_hex"],
            }
        )
    write_json(
        FIXTURES / "cpu_q5_k_authority.json",
        {
            "schema_version": 1,
            "authority": {
                "implementation": "Quartz scalar decode_q5_k / dot_q5_k",
                "revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
            },
            "activation": "a[i] = (((i * 37) % 101) - 50) / 32, rounded to FP32",
            "accumulation": "left-to-right FP32 multiply then FP32 add",
            "cases": cases,
        },
    )


def generate_matvec_tolerance() -> None:
    q4 = bytes.fromhex(
        json.loads((FIXTURES / "quant_authority.json").read_text())["cases"][0]["block_hex"]
    )
    q5 = bytes.fromhex(
        json.loads((FIXTURES / "cpu_q5_k_authority.json").read_text())["cases"][0]["block_hex"]
    )
    q6 = bytes.fromhex(
        json.loads((FIXTURES / "quant_authority.json").read_text())["cases"][2]["block_hex"]
    )
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for kind, block, columns in (
            ("q4_k", q4, 2048),
            ("q5_k", q5, 2048),
            ("q6_k", q6, 2048),
            ("q4_k", q4, 6144),
            ("q5_k", q5, 6144),
        ):
            rows = 256
            blocks_per_row = columns // 256
            payload = (block * blocks_per_row) * rows
            activation = [
                f32(((index * 37) % 101 - 50) / 32.0) for index in range(columns)
            ]
            payload_path = tmp_path / f"{kind}_{columns}.bin"
            act_path = tmp_path / f"{kind}_{columns}.act"
            payload_path.write_bytes(payload)
            act_path.write_bytes(b"".join(struct.pack("<f", value) for value in activation))
            avx = run_eval(
                "--check-matvec",
                kind,
                str(columns),
                str(rows),
                f"@{payload_path}",
                f"@{act_path}",
            )
            scalar = run_eval(
                "--check-matvec",
                kind,
                str(columns),
                str(rows),
                f"@{payload_path}",
                f"@{act_path}",
                env={"QW38_SCALAR_MATVEC": "1"},
            )
            if avx.returncode != 0:
                raise SystemExit(f"avx {kind} {columns}: {avx.stderr}")
            if scalar.returncode != 0:
                raise SystemExit(f"scalar {kind} {columns}: {scalar.stderr}")
            avx_out = bytes.fromhex(parse_output(avx.stdout)["output_f32_le_hex"])
            scalar_out = bytes.fromhex(parse_output(scalar.stdout)["output_f32_le_hex"])
            left = struct.unpack(f"<{rows}f", avx_out)
            right = struct.unpack(f"<{rows}f", scalar_out)
            errors = [abs(a - b) for a, b in zip(left, right)]
            rms = math.sqrt(sum(error * error for error in errors) / len(errors))
            cases.append(
                {
                    "kind": kind,
                    "columns": columns,
                    "rows": rows,
                    "maximum_absolute_error": max(errors),
                    "root_mean_square_error": rms,
                }
            )
            print(kind, columns, "max", max(errors), "rms", rms)
    write_json(
        FIXTURES / "cpu_avx2_matvec_tolerance.json",
        {
            "schema_version": 1,
            "path": "AVX2 Q8-activation matvec versus scalar tensor_row_dot",
            "force_scalar_environment": "QW38_SCALAR_MATVEC=1",
            "cases": cases,
        },
    )


def generate_row_samples() -> None:
    inventory = json.loads(INVENTORY.read_text())
    by_name = {tensor["name"]: tensor for tensor in inventory["tensors"]}
    cases = []
    for name, row in (
        ("token_embd.weight", 42),
        ("blk.0.attn_qkv.weight", 0),
        ("blk.0.ffn_gate.weight", 0),
    ):
        tensor = by_name[name]
        columns, rows = tensor["shape"]
        row_bytes = tensor["storage_bytes"] // rows
        with MODEL.open("rb") as model:
            model.seek(tensor["absolute_offset"] + row * row_bytes)
            payload = model.read(row_bytes)
        result = run_eval("--check-tensor-row", str(MODEL), name, str(row))
        if result.returncode != 0:
            raise SystemExit(f"{name}: {result.stderr}")
        actual = parse_output(result.stdout)
        cases.append(
            {
                "name": name,
                "row": row,
                "dtype": tensor["dtype"],
                "columns": columns,
                "rows": rows,
                "row_bytes": row_bytes,
                "row_sha256": hashlib.sha256(payload).hexdigest(),
                "dot_f32_le_hex": actual["dot_f32_le_hex"],
            }
        )
    write_json(
        FIXTURES / "cpu_tensor_rows.json",
        {
            "schema_version": 1,
            "model_sha256": inventory["model_sha256"],
            "cases": cases,
        },
    )


def generate_tokenizer() -> None:
    cases = []
    for name, text in (
        ("empty", ""),
        ("ascii", "hello world"),
        ("contractions", "I'm sure we've tested Qwen's tokenizer."),
        ("cjk", "你好，世界"),
    ):
        result = run_eval("--tokenize-hex", str(MODEL), text.encode("utf-8").hex())
        if result.returncode != 0:
            raise SystemExit(f"tokenize {name}: {result.stderr}")
        ids = (
            []
            if not result.stdout.strip()
            else [int(token) for token in result.stdout.strip().split(",")]
        )
        cases.append({"name": name, "utf8_hex": text.encode("utf-8").hex(), "ids": ids})
    write_json(
        FIXTURES / "cpu_tokenizer_2b.json",
        {"schema_version": 1, "model": "Qwen3.5-2B-Q4_K_M", "cases": cases},
    )


def generate_bind() -> None:
    result = run_eval("--check-weight-binding", str(MODEL), "valid")
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    write_json(
        FIXTURES / "cpu_weight_binding_2b.json",
        {"schema_version": 1, "fields": parse_output(result.stdout)},
    )


def generate_real_step(flag: str, fixture_name: str) -> None:
    result = run_eval(flag, str(MODEL), "valid", env={"QW38_SCALAR_MATVEC": "1"})
    if result.returncode != 0:
        raise SystemExit(f"{flag}: {result.stderr}")
    write_json(
        FIXTURES / fixture_name,
        {"schema_version": 1, "mode": "valid", "fields": parse_output(result.stdout)},
    )


def main() -> None:
    if not MODEL.exists():
        raise SystemExit("pinned 2B GGUF is missing")
    generate_q5()
    generate_matvec_tolerance()
    generate_row_samples()
    generate_tokenizer()
    generate_bind()
    generate_real_step("--check-real-gdn-step", "cpu_real_gdn_step_2b.json")
    generate_real_step("--check-real-attention-step", "cpu_real_attention_step_2b.json")
    generate_real_step("--check-real-ffn-step", "cpu_real_ffn_step_2b.json")
    generate_real_step("--check-real-scalar-token", "cpu_real_scalar_token_2b.json")


if __name__ == "__main__":
    main()
