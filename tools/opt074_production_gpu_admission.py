"""OPT-074 production-shape llama GPU numerical admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.production_numerics import (  # noqa: E402
    GGUF_SHA,
    LLAMA_REV,
    STRICT_CUD001_ABS,
    STRICT_CUD001_RMS,
)
from tools.production_numerics_v2 import (  # noqa: E402
    CALIBRATION_LAYERS,
    FAMILY_SHAPES,
    HELD_OUT_LAYERS,
    MAX_PRODUCTION_FP64_DOTS,
    MAX_ROWS,
    STAGING_LLAMA_SUM_X,
    STAGING_Q8_FP32,
    STAGING_QUARTZ_SUM_Q,
    compare_metrics,
    freeze_case_ids,
    v2_abs_rms_ceiling,
    v2_cosine_ceiling,
)
from tools.run_optimization_task import loop_product  # noqa: E402

CONTRACT = ROOT / "pins/opt074_production_gpu_admission_contract.json"
ITERATION = ROOT / "pins/opt074_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt074_production_gpu_admission.json"
REPORT = ROOT / "evidence/optimization/opt074-production-gpu-admission/REPORT.md"
INVENTORY = ROOT / "pins/tensor_inventory.json"
OPT059_MANIFEST = ROOT / "pins/opt059_admission_manifest.json"
OPT059_FIXTURE = ROOT / "fixtures/opt059_gpu_numerics.json"
OPT059_CONTRACT = ROOT / "pins/production_numerics_v2_contract.json"
MODEL = ROOT / "models/Qwen3.8-27B-Q4_K_M.gguf"
EXPORT_BIN = ROOT / ".cache/authorities/llama-build/bin/qw38-llama-projection-export"
CAPTURE_BIN = ROOT / "build/qw38-cuda-opt043-activation-capture-test"
NATIVE_OPT059 = ROOT / "build/qw38-cuda-opt059-numerics-test"
CACHE_DIR = ROOT / ".cache/opt074-captures"
EVIDENCE_DIR = ROOT / "evidence/optimization/opt074-production-gpu-admission"
LLAMA_IMAGE = "qw38-llama-authority:cuda-13.0.2"
CUDA_IMAGE = "qw38-cuda:13.0.2"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
PRODUCER = "qw38-llama-projection-export/opt074"
Q4K_BLOCK = 144
Q6K_BLOCK = 210
Q80_BLOCK = 34
Q4K_VALUES = 256
Q80_VALUES = 32

PHASES = ("q4-gate-up", "q4-down", "q4", "q8-q6")
SUM_VARIANTS = {
    "sum_q": STAGING_QUARTZ_SUM_Q,
    "sum_x": STAGING_LLAMA_SUM_X,
    "exact_integer_sum": "exact_integer_sum_recomputed_dp4a",
}
CONSUMERS = {
    STAGING_QUARTZ_SUM_Q: "quartz_q8_1_integer_dot",
    STAGING_LLAMA_SUM_X: "llama_gpu_q8_1_mmvq",
    STAGING_Q8_FP32: "packed_fp32_q4_mmv",
    "exact_integer_sum_recomputed_dp4a": "pinned_q4_mmv_recomputes_integer_sum",
}


class AdmissionError(AssertionError):
    """Fail-closed OPT-074 admission error."""


Exporter = Callable[[Sequence[str]], dict[str, Any]]
CaptureFn = Callable[[int, Path], dict[str, Any]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_floats(values: Sequence[float]) -> str:
    return sha256_bytes(b"".join(struct.pack("<f", float(item)) for item in values))


def bf16_to_f32(bits: int) -> float:
    raw = struct.pack("<I", (int(bits) & 0xFFFF) << 16)
    return struct.unpack("<f", raw)[0]


def load_bf16(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 2 != 0:
        raise AdmissionError(f"odd BF16 file {path}")
    return list(struct.unpack("<" + "H" * (len(data) // 2), data))


def load_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise AdmissionError(f"odd FP32 file {path}")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def write_bf16(path: Path, bits: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<" + "H" * len(bits), *bits))


def write_f32(path: Path, values: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<" + "f" * len(values), *[float(v) for v in values]))


def half_to_float(bits: int) -> float:
    sign = (bits & 0x8000) << 16
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x03FF
    if exponent == 0:
        if fraction == 0:
            packed = sign
        else:
            shifts = 0
            while (fraction & 0x0400) == 0:
                fraction <<= 1
                shifts += 1
            fraction &= 0x03FF
            packed = sign | ((113 - shifts) << 23) | (fraction << 13)
    elif exponent == 0x1F:
        packed = sign | 0x7F800000 | (fraction << 13)
    else:
        packed = sign | ((exponent + 112) << 23) | (fraction << 13)
    return struct.unpack(">f", struct.pack(">I", packed))[0]


def layer_kind(layer: int) -> str:
    return "attention" if int(layer) % 4 == 3 else "gdn"


def inventory_tensors(
    inventory: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    data = inventory or load_json(INVENTORY)
    return [dict(row) for row in data["tensors"]]


def find_tensor(
    name: str, inventory: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    for tensor in inventory_tensors(inventory):
        if tensor["name"] == name:
            return dict(tensor)
    return None


def tensor_name_for_case(case: Mapping[str, Any]) -> str | None:
    role = str(case["role"])
    layer = int(case["layer"])
    kind = layer_kind(layer)
    if role == "ffn_gate_up":
        return f"blk.{layer}.ffn_gate.weight"
    if role == "ffn_down":
        return f"blk.{layer}.ffn_down.weight"
    if role == "vocab":
        return "output.weight"
    if role == "mixer_qkv":
        if kind != "gdn":
            return None
        return f"blk.{layer}.attn_qkv.weight"
    if role == "mixer_output":
        if kind != "gdn":
            return None
        return f"blk.{layer}.ssm_out.weight"
    raise AdmissionError(f"unknown role {role}")


def up_tensor_name(case: Mapping[str, Any]) -> str | None:
    if str(case["role"]) != "ffn_gate_up":
        return None
    return f"blk.{int(case['layer'])}.ffn_up.weight"


def require_inventory_match(case: Mapping[str, Any], tensor: Mapping[str, Any]) -> None:
    columns, rows = [int(part) for part in tensor["shape"]]
    if columns != int(case["columns"]) or rows != int(case["rows"]):
        raise AdmissionError(
            f"shape mismatch for {tensor['name']}: inventory {tensor['shape']} "
            f"vs case K={case['columns']} M={case['rows']}"
        )
    expected = {
        "q4_k": "Q4_K",
        "q8_0": "Q8_0",
        "q6_k": "Q6_K",
    }[str(case["family"])]
    if str(tensor["dtype"]) != expected:
        raise AdmissionError(
            f"dtype mismatch for {tensor['name']}: {tensor['dtype']} != {expected}"
        )


def sample_rows(rows: int, limit: int = MAX_ROWS) -> list[int]:
    total = int(rows)
    if total <= 0:
        raise AdmissionError("rows must be positive")
    if total <= limit:
        return list(range(total))
    half = limit // 2
    head = list(range(half))
    tail = list(range(total - (limit - half), total))
    return head + tail


def require_rows(got: Sequence[int], expected: Sequence[int]) -> None:
    if list(got) != list(expected):
        raise AdmissionError(f"missing rows: got {list(got)} expected {list(expected)}")


def require_dispatch(record: Mapping[str, Any], *, rows: int, columns: int) -> None:
    if int(record.get("row_limit", 0) or 0) != 0:
        raise AdmissionError("wrong dispatch: row-limit is not production evidence")
    if int(record.get("n", 0)) != 1:
        raise AdmissionError("wrong dispatch: N must be 1")
    if int(record.get("full_m", -1)) != int(rows):
        raise AdmissionError("wrong dispatch: full M was not preserved")
    if int(record.get("ne1", -1)) != int(rows):
        raise AdmissionError("wrong dispatch: exported M changed")
    if int(record.get("ne0", -1)) != int(columns):
        raise AdmissionError(f"wrong K {record.get('ne0')} != {columns}")
    if str(record.get("dispatch_family")) != "mmvq":
        raise AdmissionError(f"wrong dispatch family {record.get('dispatch_family')!r}")
    if record.get("not_cpu_replica") is not True:
        raise AdmissionError("authority must be llama GPU, not a CPU replica")


def require_authority(
    record: Mapping[str, Any],
    *,
    revision: str = LLAMA_REV,
    gguf_sha: str = GGUF_SHA,
    producer: str = PRODUCER,
) -> None:
    if str(record.get("llama_revision")) != revision:
        raise AdmissionError("authority revision mutated")
    got_sha = str(record.get("gguf_sha256") or "")
    if got_sha and got_sha != gguf_sha:
        raise AdmissionError("GGUF identity mutated")
    if str(record.get("producer")) != producer:
        raise AdmissionError("producer tag mutated")


def require_finite(values: Sequence[float], *, label: str) -> None:
    bad = sum(1 for value in values if not math.isfinite(float(value)))
    if bad:
        raise AdmissionError(f"nonfinite {label}: {bad}")


def reference_cache_key(
    *,
    tensor: str,
    columns: int,
    rows: int,
    input_sha: str,
    fused: bool,
    producer: str = PRODUCER,
    revision: str = LLAMA_REV,
    gguf_sha: str = GGUF_SHA,
) -> str:
    payload = json.dumps(
        {
            "task": "OPT-074",
            "llama_revision": revision,
            "gguf_sha256": gguf_sha,
            "producer": producer,
            "tensor": tensor,
            "columns": int(columns),
            "rows": int(rows),
            "n": 1,
            "row_limit": 0,
            "input_sha": input_sha,
            "fused_gate_up_swiglu": bool(fused),
            "dispatch_family": "mmvq",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("ascii"))


def capture_identity(*, token_position: int, gguf_sha: str = GGUF_SHA) -> str:
    payload = json.dumps(
        {
            "gguf_sha": gguf_sha,
            "token_generator": TOKEN_GENERATOR,
            "token_position": int(token_position),
            "layers": list(CALIBRATION_LAYERS + HELD_OUT_LAYERS),
            "kinds": [
                "ffn_preprojection",
                "mixer_preprojection",
                "swiglu_down_input",
                "gdn_or_attn_mix_6144",
                "final_norm",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("ascii"))


def activation_stem(case: Mapping[str, Any]) -> str:
    layer = int(case["layer"])
    name = str(case["activation"])
    if name == "final_norm":
        return "final_norm"
    return f"L{layer}_{name}"


def _q4_scale_min(packed: bytes, index: int) -> tuple[int, int]:
    if index < 4:
        return packed[index] & 63, packed[index + 4] & 63
    scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4)
    minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)
    return scale, minimum


def decode_q4_k_block(block: bytes) -> list[float]:
    d = half_to_float(block[0] | (block[1] << 8))
    dmin = half_to_float(block[2] | (block[3] << 8))
    packed = block[4:16]
    quants = block[16:144]
    output = [0.0] * Q4K_VALUES
    offset = 0
    for group in range(4):
        low_s, low_m = _q4_scale_min(packed, group * 2)
        high_s, high_m = _q4_scale_min(packed, group * 2 + 1)
        low_d = d * float(low_s)
        low_min = dmin * float(low_m)
        high_d = d * float(high_s)
        high_min = dmin * float(high_m)
        for lane in range(32):
            quant = quants[group * 32 + lane]
            output[offset + lane] = low_d * float(quant & 15) - low_min
            output[offset + 32 + lane] = high_d * float(quant >> 4) - high_min
        offset += 64
    return output


def decode_q6_k_block(block: bytes) -> list[float]:
    low = block[0:128]
    high = block[128:192]
    scales = block[192:208]
    d = half_to_float(block[208] | (block[209] << 8))
    output = [0.0] * Q4K_VALUES
    for half in range(2):
        low_offset = half * 64
        high_offset = half * 32
        scale_offset = half * 8
        output_offset = half * 128
        for lane in range(32):
            scale_pair = lane // 16
            q1 = (
                (low[low_offset + lane] & 15) | ((high[high_offset + lane] & 3) << 4)
            ) - 32
            q2 = (
                (low[low_offset + lane + 32] & 15)
                | (((high[high_offset + lane] >> 2) & 3) << 4)
            ) - 32
            q3 = (
                (low[low_offset + lane] >> 4)
                | (((high[high_offset + lane] >> 4) & 3) << 4)
            ) - 32
            q4 = (
                (low[low_offset + lane + 32] >> 4)
                | (((high[high_offset + lane] >> 6) & 3) << 4)
            ) - 32
            quants = (q1, q2, q3, q4)
            for group, quant in enumerate(quants):
                scale_byte = scales[scale_offset + scale_pair + group * 2]
                scale = scale_byte if scale_byte < 128 else scale_byte - 256
                output[output_offset + lane + group * 32] = (
                    d * float(scale) * float(quant)
                )
    return output


def decode_q8_0_block(block: bytes) -> list[float]:
    scale = half_to_float(block[0] | (block[1] << 8))
    values = []
    for index in range(Q80_VALUES):
        byte = block[index + 2]
        quant = byte if byte < 128 else byte - 256
        values.append(scale * float(quant))
    return values


_MODEL_FILE = None
_MODEL_MMAP: mmap.mmap | None = None


def model_bytes() -> memoryview:
    global _MODEL_FILE, _MODEL_MMAP
    if _MODEL_MMAP is None:
        _MODEL_FILE = MODEL.open("rb")
        _MODEL_MMAP = mmap.mmap(_MODEL_FILE.fileno(), 0, access=mmap.ACCESS_READ)
    return memoryview(_MODEL_MMAP)


def mmap_tensor(path: Path, tensor: Mapping[str, Any]) -> memoryview:
    if path.resolve() != MODEL.resolve():
        raise AdmissionError(f"unexpected model path {path}")
    start = int(tensor["absolute_offset"])
    size = int(tensor["storage_bytes"])
    return model_bytes()[start : start + size]


def decode_row(family: str, blob: memoryview, row: int, columns: int) -> list[float]:
    if family == "q4_k":
        blocks = columns // Q4K_VALUES
        start = row * blocks * Q4K_BLOCK
        values: list[float] = []
        for block in range(blocks):
            offset = start + block * Q4K_BLOCK
            values.extend(decode_q4_k_block(bytes(blob[offset : offset + Q4K_BLOCK])))
        return values
    if family == "q6_k":
        blocks = columns // Q4K_VALUES
        start = row * blocks * Q6K_BLOCK
        values = []
        for block in range(blocks):
            offset = start + block * Q6K_BLOCK
            values.extend(decode_q6_k_block(bytes(blob[offset : offset + Q6K_BLOCK])))
        return values
    if family == "q8_0":
        blocks = columns // Q80_VALUES
        start = row * blocks * Q80_BLOCK
        values = []
        for block in range(blocks):
            offset = start + block * Q80_BLOCK
            values.extend(decode_q8_0_block(bytes(blob[offset : offset + Q80_BLOCK])))
        return values
    raise AdmissionError(f"unsupported family {family}")


def fp64_dot(weight_row: Sequence[float], activation: Sequence[float]) -> float:
    if len(weight_row) != len(activation):
        raise AdmissionError("FP64 K mismatch")
    total = 0.0
    for left, right in zip(weight_row, activation, strict=True):
        total += float(left) * float(right)
    return total


def parse_export_payload(text: str) -> dict[str, Any]:
    for prefix in (
        "QW38_OPT074_LLAMA_GPU_EXPORT=",
        "QW38_OPT059_LLAMA_GPU_EXPORT=",
    ):
        for line in text.splitlines():
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :])
    raise AdmissionError("missing llama GPU export payload")


def frozen_cases() -> dict[str, Any]:
    return freeze_case_ids()


def historical_opt059_preserved() -> bool:
    manifest = load_json(OPT059_MANIFEST)
    fixture = load_json(OPT059_FIXTURE)
    contract = load_json(OPT059_CONTRACT)
    return (
        fixture.get("v2_admitted") is not True
        and any(
            entry.get("v2_admitted") is False for entry in manifest.get("entries", [])
        )
        and contract.get("task") == "OPT-059"
        and any(
            int(entry.get("columns", -1)) == 0 for entry in manifest.get("entries", [])
        )
    )


def cases_for_phase(phase: str, *, feedback: bool) -> list[dict[str, Any]]:
    frozen = frozen_cases()
    calibration = list(frozen["calibration"])
    held = list(frozen["held_out"])
    if phase == "q4-gate-up":
        selected = [
            row
            for row in calibration
            if row["role"] == "ffn_gate_up" and int(row["layer"]) == 0
        ]
        if feedback:
            return selected[:1]
        return selected
    if phase == "q4-down":
        selected = [
            row
            for row in calibration
            if row["role"] == "ffn_down" and int(row["layer"]) == 0
        ]
        if feedback:
            return selected[:1]
        return selected
    if phase == "q4":
        roles = {"ffn_gate_up", "ffn_down"}
        return [row for row in calibration + held if row["role"] in roles]
    if phase == "q8-q6":
        roles = {"mixer_qkv", "mixer_output", "vocab"}
        return [row for row in calibration + held if row["role"] in roles]
    raise AdmissionError(f"unknown phase {phase}")


def family_id(case: Mapping[str, Any]) -> str:
    return str(case["id"]).rsplit("_L", 1)[0]


def authenticate_model(path: Path = MODEL) -> str:
    cache = CACHE_DIR / "gguf.sha256"
    if cache.is_file():
        recorded = cache.read_text(encoding="utf-8").strip()
        if recorded == GGUF_SHA and path.is_file():
            return recorded
    if not path.is_file():
        raise AdmissionError(f"missing model {path}")
    digest = sha256_file(path)
    if digest != GGUF_SHA:
        raise AdmissionError("GGUF SHA-256 does not match the pinned artifact")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(digest + "\n", encoding="utf-8")
    return digest


def to_workspace(part: str) -> str:
    if part.startswith(str(ROOT)):
        return "/workspace/" + Path(part).resolve().relative_to(ROOT).as_posix()
    return part


def docker_run(image: str, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    wrapped = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        image,
        *[to_workspace(str(part)) for part in command],
    ]
    return subprocess.run(
        wrapped, cwd=ROOT, check=False, capture_output=True, text=True
    )


def default_exporter(command: Sequence[str]) -> dict[str, Any]:
    completed = docker_run(LLAMA_IMAGE, command)
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        raise AdmissionError(
            f"exporter failed rc={completed.returncode}: {completed.stderr}"
        )
    return parse_export_payload(text)


def default_capture(token_position: int, dump_dir: Path) -> dict[str, Any]:
    dump_dir.mkdir(parents=True, exist_ok=True)
    if not CAPTURE_BIN.is_file():
        raise AdmissionError(f"missing capture binary {CAPTURE_BIN}")
    command = [
        str(CAPTURE_BIN),
        str(MODEL),
        f"t{int(token_position)}",
        "--dump-dir",
        str(dump_dir),
    ]
    completed = docker_run(CUDA_IMAGE, command)
    if completed.returncode != 0:
        raise AdmissionError(
            f"capture failed rc={completed.returncode}: {completed.stderr}"
        )
    return {"stdout": completed.stdout, "stderr": completed.stderr}


def ensure_capture(
    token_position: int,
    *,
    capture_fn: CaptureFn | None = None,
) -> Path:
    identity = capture_identity(token_position=token_position)
    dump_dir = CACHE_DIR / identity / f"t{int(token_position)}"
    marker = dump_dir / "bundle.json"
    if marker.is_file() and (dump_dir / "final_norm.bf16").is_file():
        return dump_dir
    (capture_fn or default_capture)(token_position, dump_dir)
    dump_json(
        marker,
        {
            "capture_key": identity,
            "token_position": int(token_position),
            "token_generator": TOKEN_GENERATOR,
            "gguf_sha256": GGUF_SHA,
            "llama_revision": LLAMA_REV,
        },
    )
    if not (dump_dir / "final_norm.bf16").is_file():
        raise AdmissionError(f"capture missing final_norm at t{token_position}")
    return dump_dir


def load_activation(dump_dir: Path, case: Mapping[str, Any]) -> list[int]:
    stem = activation_stem(case)
    path = dump_dir / f"{stem}.bf16"
    if not path.is_file():
        raise AdmissionError(f"absent capture {stem} for {case['id']} at {dump_dir}")
    bits = load_bf16(path)
    if len(bits) != int(case["columns"]):
        raise AdmissionError(
            f"capture K {len(bits)} != {case['columns']} for {case['id']}"
        )
    return bits


def export_projection(
    *,
    tensor: str,
    case: Mapping[str, Any],
    input_bf16: Path,
    output: Path,
    retain_bf16: Path,
    q81: Path | None,
    fused: bool,
    up_tensor: str | None,
    exporter: Exporter | None = None,
) -> dict[str, Any]:
    if not EXPORT_BIN.is_file() and exporter is None:
        raise AdmissionError(f"missing exporter {EXPORT_BIN}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(EXPORT_BIN),
        "--evidence",
        "--task",
        "OPT-074",
        "--model",
        str(MODEL),
        "--tensor",
        tensor,
        "--input-bf16",
        str(input_bf16),
        "--retain-bf16",
        str(retain_bf16),
        "--output",
        str(output),
        "--producer",
        PRODUCER,
        "--gguf-sha",
        GGUF_SHA,
    ]
    if q81 is not None:
        command.extend(["--q8-1", str(q81)])
    if fused:
        if up_tensor is None:
            raise AdmissionError("fused export needs up tensor")
        command.extend(["--fuse-swiglu", "--up-tensor", up_tensor])
    payload = (exporter or default_exporter)(command)
    require_authority(payload)
    require_dispatch(
        payload,
        rows=int(case["rows"]) if not fused else int(case["rows"]),
        columns=int(case["columns"]),
    )
    if "--row-limit" in command:
        raise AdmissionError("wrong dispatch: exporter invoked with row-limit")
    return payload


def freeze_family_ceilings(
    calibration_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not calibration_metrics:
        raise AdmissionError("no calibration metrics to freeze")
    abs_err = max(float(row["max_abs"]) for row in calibration_metrics)
    rms_err = max(float(row["rms"]) for row in calibration_metrics)
    cosine = max(float(row["one_minus_cosine"]) for row in calibration_metrics)
    nonfinite = sum(int(row.get("nonfinite", 0)) for row in calibration_metrics)
    return {
        "abs": v2_abs_rms_ceiling(STRICT_CUD001_ABS, abs_err, nonfinite=nonfinite),
        "rms": v2_abs_rms_ceiling(STRICT_CUD001_RMS, rms_err, nonfinite=nonfinite),
        "one_minus_cosine": v2_cosine_ceiling(cosine, nonfinite=nonfinite),
        "frozen_before_candidates": True,
        "held_out_cannot_enlarge": True,
    }


def held_out_validates(ceilings: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    if int(metrics.get("nonfinite", 0)) != 0:
        return False
    return (
        float(metrics["max_abs"]) <= float(ceilings["abs"]["ceiling"])
        and float(metrics["rms"]) <= float(ceilings["rms"]["ceiling"])
        and float(metrics["one_minus_cosine"])
        <= float(ceilings["one_minus_cosine"]["ceiling"])
    )


def compare_llama_to_fp64(
    llama: Sequence[float],
    fp64: Sequence[float],
) -> dict[str, Any]:
    require_finite(llama, label="llama_gpu")
    require_finite(fp64, label="fp64")
    return compare_metrics([float(v) for v in llama], [float(v) for v in fp64])


def evaluate_case(
    case: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any],
    decoded_rows: dict[tuple[str, int], list[float]],
    blob_cache: dict[str, memoryview],
    exporter: Exporter | None,
    capture_fn: CaptureFn | None,
    run_dir: Path,
) -> dict[str, Any]:
    name = tensor_name_for_case(case)
    base = {
        "id": case["id"],
        "family": case.get("family"),
        "family_id": family_id(case),
        "role": case["role"],
        "split": case.get("split"),
        "layer": case["layer"],
        "layer_kind": layer_kind(int(case["layer"])),
        "token_position": case.get("token_position"),
        "substituted": False,
    }
    if name is None:
        return {
            **base,
            "status": "absent_layer_kind",
            "reason": "inventory has no matching Q8 mixer tensor for this layer kind",
        }
    tensor = find_tensor(name, inventory)
    if tensor is None:
        return {**base, "status": "absent_tensor", "tensor": name}
    require_inventory_match(case, tensor)
    dump_dir = ensure_capture(int(case["token_position"]), capture_fn=capture_fn)
    try:
        bf16 = load_activation(dump_dir, case)
    except AdmissionError as exc:
        return {**base, "status": "absent_capture", "reason": str(exc)}
    input_sha = sha256_bytes(struct.pack("<" + "H" * len(bf16), *bf16))
    cache_key = reference_cache_key(
        tensor=name,
        columns=int(case["columns"]),
        rows=int(case["rows"]),
        input_sha=input_sha,
        fused=False,
    )
    work = CACHE_DIR / "refs" / cache_key
    work.mkdir(parents=True, exist_ok=True)
    (run_dir / "refs" / str(case["id"])).mkdir(parents=True, exist_ok=True)
    input_path = work / "activation.bf16"
    retain_path = work / "retained.bf16"
    output_path = work / "llama_gpu.f32"
    q81_path = work / "q8_1_sum_x.bin"
    write_bf16(input_path, bf16)
    fused = str(case["role"]) == "ffn_gate_up"
    cached = work / "record.json"
    if cached.is_file():
        record = load_json(cached)
        if record.get("reference_cache_key") == cache_key and output_path.is_file():
            llama = load_f32(output_path)
            payload = dict(record)
            payload["cache_hit"] = True
        else:
            payload = None
            llama = []
    else:
        payload = None
        llama = []
    if payload is None:
        payload = export_projection(
            tensor=name,
            case=case,
            input_bf16=input_path,
            output=output_path,
            retain_bf16=retain_path,
            q81=q81_path,
            fused=False,
            up_tensor=up_tensor_name(case),
            exporter=exporter,
        )
        llama = load_f32(output_path)
        payload["reference_cache_key"] = cache_key
        payload["cache_hit"] = False
        dump_json(cached, payload)
    require_finite(llama, label=f"llama[{case['id']}]")
    if len(llama) != int(case["rows"]):
        raise AdmissionError(
            f"{case['id']} exported {len(llama)} rows, expected {case['rows']}"
        )
    rows = sample_rows(int(case["rows"]))
    require_rows(rows, sample_rows(int(case["rows"])))
    if name not in blob_cache:
        blob_cache[name] = mmap_tensor(MODEL, tensor)
    activation = [bf16_to_f32(bits) for bits in bf16]
    fp64_vals: list[float] = []
    llama_sample: list[float] = []
    dots = 0
    for row in rows:
        key = (name, row)
        if key not in decoded_rows:
            decoded_rows[key] = decode_row(
                str(case["family"]), blob_cache[name], row, int(case["columns"])
            )
        fp64_vals.append(fp64_dot(decoded_rows[key], activation))
        llama_sample.append(llama[row])
        dots += 1
        if dots > MAX_PRODUCTION_FP64_DOTS:
            raise AdmissionError("fp64 dot budget exceeded")
    metrics = compare_llama_to_fp64(llama_sample, fp64_vals)
    result = {
        "id": case["id"],
        "status": "measured",
        "family": case["family"],
        "family_id": family_id(case),
        "role": case["role"],
        "split": case["split"],
        "layer": case["layer"],
        "layer_kind": layer_kind(int(case["layer"])),
        "token_position": case["token_position"],
        "tensor": name,
        "columns": case["columns"],
        "rows": case["rows"],
        "sampled_rows": rows,
        "fp64_dots": dots,
        "input_sha256": input_sha,
        "reference_cache_key": cache_key,
        "dispatch": {
            "family": payload.get("dispatch_family"),
            "n": payload.get("n"),
            "m": payload.get("ne1"),
            "full_m": payload.get("full_m"),
            "row_limit": payload.get("row_limit"),
            "cuda_synchronized": payload.get("cuda_synchronized", True),
        },
        "llama_revision": payload.get("llama_revision"),
        "producer": payload.get("producer"),
        "metrics": metrics,
        "q8_1": payload.get("q8_1"),
        "fused_export": False,
        "cache_hit": payload.get("cache_hit", False),
        "substituted": False,
    }
    if fused:
        fused_out = work / "llama_gpu_fused.f32"
        fused_key = reference_cache_key(
            tensor=name,
            columns=int(case["columns"]),
            rows=int(case["rows"]),
            input_sha=input_sha,
            fused=True,
        )
        fused_record = work / "fused.json"
        if (
            fused_record.is_file()
            and load_json(fused_record).get("reference_cache_key") == fused_key
            and fused_out.is_file()
        ):
            result["fused_export"] = True
            result["fused_cache_hit"] = True
        else:
            fused_payload = export_projection(
                tensor=name,
                case=case,
                input_bf16=input_path,
                output=fused_out,
                retain_bf16=retain_path,
                q81=None,
                fused=True,
                up_tensor=up_tensor_name(case),
                exporter=exporter,
            )
            fused_payload["reference_cache_key"] = fused_key
            dump_json(fused_record, fused_payload)
            result["fused_export"] = True
            result["fused_dispatch"] = fused_payload.get("dispatch_family")
            result["fused_gate_up_swiglu"] = fused_payload.get("fused_gate_up_swiglu")
    return result


def expected_family_ids(ident: str, split: str) -> list[str]:
    frozen = frozen_cases()
    ids = []
    for row in frozen[split]:
        if family_id(row) != ident:
            continue
        if tensor_name_for_case(row) is None:
            continue
        ids.append(str(row["id"]))
    return ids


def build_admission_table(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row.get("family_id", family_id(row))), []).append(row)
    table = []
    idents = {str(shape["id"]) for shape in FAMILY_SHAPES}
    idents.update(grouped)
    for ident in sorted(idents):
        rows = list(grouped.get(ident, []))
        measured = [row for row in rows if row.get("status") == "measured"]
        calib = [row for row in measured if row.get("split") == "calibration"]
        held = [row for row in measured if row.get("split") == "held_out"]
        absent = [
            row
            for row in rows
            if row.get("status")
            in {"absent_layer_kind", "absent_tensor", "absent_capture"}
        ]
        expected_calib = expected_family_ids(ident, "calibration")
        expected_held = expected_family_ids(ident, "held_out")
        calib_ids = {str(row["id"]) for row in calib}
        held_ids = {str(row["id"]) for row in held}
        complete = set(expected_calib) <= calib_ids and set(expected_held) <= held_ids
        if not calib:
            table.append(
                {
                    "id": ident,
                    "v2_admitted": False,
                    "coverage": "unadmitted",
                    "wildcard_columns": False,
                    "complete": False,
                    "reason": "missing_production_gpu_outputs",
                    "expected_calibration": expected_calib,
                    "expected_held_out": expected_held,
                    "absent": [row.get("id") for row in absent],
                }
            )
            continue
        ceilings = freeze_family_ceilings([row["metrics"] for row in calib])
        held_ok = bool(held) and all(
            held_out_validates(ceilings, row["metrics"]) for row in held
        )
        admitted = bool(complete and ceilings["abs"]["admitted"] and held_ok)
        table.append(
            {
                "id": ident,
                "family": calib[0]["family"],
                "columns": calib[0]["columns"],
                "rows": calib[0]["rows"],
                "wildcard_columns": False,
                "v2_admitted": admitted,
                "coverage": "admitted" if admitted else "unadmitted",
                "complete": complete,
                "calibration_cases": [row["id"] for row in calib],
                "held_out_cases": [row["id"] for row in held],
                "held_out_validates": held_ok,
                "held_out_cannot_enlarge": True,
                "reason": None
                if admitted
                else (
                    "held_out_exceeds_frozen_calibration_ceiling"
                    if complete and not held_ok
                    else "incomplete_or_unadmitted"
                ),
                "ceilings": ceilings,
                "max_llama_abs": max(float(row["metrics"]["max_abs"]) for row in calib),
                "max_held_out_abs": max(
                    (float(row["metrics"]["max_abs"]) for row in held), default=None
                ),
                "staging": STAGING_Q8_FP32
                if calib[0]["family"] == "q4_k"
                else STAGING_LLAMA_SUM_X,
                "sum_variants": dict(SUM_VARIANTS),
                "consumers": dict(CONSUMERS),
                "absent": [row.get("id") for row in absent],
            }
        )
    return table


def write_report(fixture: Mapping[str, Any]) -> None:
    table = fixture.get("admission_table", [])
    lines = [
        "# OPT-074 production-shape llama GPU numerical admission",
        "",
        f"Status: **{fixture.get('status')}**. "
        "`claims_performance_improvement: false`. Authority is pinned llama.cpp "
        f"`{LLAMA_REV}` and GGUF SHA-256 `{GGUF_SHA}`.",
        "",
        "OPT-059 v1/v2 historical missing-evidence records are preserved. This "
        "sitting exports full-M N=1 llama GPU projections on captured BF16 "
        "activations. `--row-limit` is not production evidence.",
        "",
        "## Frozen OPT-059 cases",
        "",
        "Calibration layers 0/3/31/32 at tokens 128/512. Held-out layers 62/63 "
        "at 2048/4095. Host checks sample 16 rows; dispatch uses full M. Held-out "
        "validates frozen calibration ceilings and cannot enlarge them.",
        "",
        "## Admission table",
        "",
        "| Family | K | M | v2 admitted | coverage | complete | held-out validates | calib abs | held-out abs |",
        "|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row.get('id')} | {row.get('columns', '')} | {row.get('rows', '')} | "
            f"{row.get('v2_admitted')} | {row.get('coverage')} | {row.get('complete', '')} | "
            f"{row.get('held_out_validates', '')} | {row.get('max_llama_abs', '')} | "
            f"{row.get('max_held_out_abs', '')} |"
        )
    lines.extend(
        [
            "",
            "## Dispatch and staging",
            "",
            "Quantized N=1 uses llama CUDA `mmvq` (`ggml_cuda_should_use_mmvq` on "
            "Blackwell, ne11=1). CUDA backend synchronize plus "
            "`cudaDeviceSynchronize` before host reads. Q8_1 bytes come from the "
            "private adapter `quantize_q8_1_sum_x` (`half(sum(x))`), not "
            "unsupported `ggml_cpy` F32→Q8_1. Quartz `sum_q`, llama `sum_x`, and "
            "exact integer sums (Q4 MMV recomputes) stay distinct consumers.",
            "",
            "## Proof limit",
            "",
        ]
    )
    for item in fixture.get("proof_limit", []):
        lines.append(f"- {item}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase(
    phase: str,
    run_dir: Path | None = None,
    *,
    exporter: Exporter | None = None,
    capture_fn: CaptureFn | None = None,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    destination = Path(run_dir) if run_dir is not None else EVIDENCE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    inventory = load_json(INVENTORY)
    feedback = phase in {"q4-gate-up", "q4-down"}
    selected = cases_for_phase(phase, feedback=feedback)
    results: list[dict[str, Any]] = []
    decoded_rows: dict[tuple[str, int], list[float]] = {}
    blob_cache: dict[str, memoryview] = {}
    gpu = not skip_gpu
    if gpu:
        authenticate_model()
        if not EXPORT_BIN.is_file() and exporter is None:
            raise AdmissionError("missing production GPU exporter")
    for case in selected:
        if skip_gpu:
            name = tensor_name_for_case(case)
            if name is None:
                results.append(
                    {
                        "id": case["id"],
                        "status": "absent_layer_kind",
                        "family_id": family_id(case),
                        "family": case["family"],
                        "role": case["role"],
                        "split": case["split"],
                        "layer": case["layer"],
                        "substituted": False,
                        "layer_kind": layer_kind(int(case["layer"])),
                    }
                )
                continue
            tensor = find_tensor(name, inventory)
            if tensor is None:
                results.append(
                    {
                        "id": case["id"],
                        "status": "absent_tensor",
                        "family_id": family_id(case),
                        "substituted": False,
                    }
                )
                continue
            require_inventory_match(case, tensor)
            results.append(
                {
                    "id": case["id"],
                    "status": "host_validated",
                    "tensor": name,
                    "family_id": family_id(case),
                    "family": case["family"],
                    "role": case["role"],
                    "split": case["split"],
                    "layer": case["layer"],
                    "columns": case["columns"],
                    "rows": case["rows"],
                    "sampled_rows": sample_rows(int(case["rows"])),
                    "substituted": False,
                }
            )
            continue
        results.append(
            evaluate_case(
                case,
                inventory=inventory,
                decoded_rows=decoded_rows,
                blob_cache=blob_cache,
                exporter=exporter,
                capture_fn=capture_fn,
                run_dir=destination,
            )
        )
    previous_cases: list[dict[str, Any]] = []
    if FIXTURE.is_file():
        previous_cases = list(load_json(FIXTURE).get("cases", []))
    merged = {str(row["id"]): dict(row) for row in previous_cases}
    for row in results:
        merged[str(row["id"])] = dict(row)
    combined = list(merged.values())
    table = build_admission_table(combined) if not skip_gpu else []
    q4_required = [row for row in table if str(row.get("family")) == "q4_k"]
    q4_measured = bool(q4_required) and all(row.get("complete") for row in q4_required)
    requested_ok = all(
        row.get("status") in {"measured", "absent_layer_kind", "host_validated"}
        for row in results
    )
    missing = any(row.get("status") == "absent_capture" for row in results)
    if skip_gpu:
        blocked = False
    elif missing or not requested_ok:
        blocked = True
    elif phase == "q4" and not q4_measured:
        blocked = True
    else:
        blocked = False
    status = (
        "host_validated"
        if skip_gpu
        else "blocked_missing_gpu"
        if blocked
        else "measured"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-074",
        "phase": phase,
        "success": not blocked,
        "status": status,
        "result_class": "blocked" if blocked else "ok",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "producer": PRODUCER,
        "opt059_historical_preserved": historical_opt059_preserved(),
        "row_limit_used_for_evidence": False,
        "full_m": True,
        "n": 1,
        "fp64_dots": sum(int(row.get("fp64_dots", 0)) for row in results),
        "cases": combined if not skip_gpu else results,
        "phase_cases": [row["id"] for row in results],
        "admission_table": table,
        "sum_variants": dict(SUM_VARIANTS),
        "consumers": dict(CONSUMERS),
        "frozen_case_ids": {
            "calibration": [row["id"] for row in frozen_cases()["calibration"]],
            "held_out": [row["id"] for row in frozen_cases()["held_out"]],
        },
        "native_reuse": [
            str(NATIVE_OPT059.relative_to(ROOT)),
            str(EXPORT_BIN.relative_to(ROOT)),
            str(CAPTURE_BIN.relative_to(ROOT)),
        ],
        "proof_limit": [
            "no throughput claim",
            "no production pin changes",
            "full-M N=1 llama GPU dispatch",
            "no --row-limit evidence",
            "held-out cannot enlarge calibration ceilings",
            "wildcard columns=0 is not production coverage",
            "absent layer-kind is explicit",
            "OPT-059 historical missing evidence preserved",
        ],
        "measurement_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_path": str(REPORT.relative_to(ROOT)),
        "loop_product_hint": loop_product(
            {"cases": max(1, len(selected)), "tokens": 1, "samples": 1}
        ),
    }
    if payload["fp64_dots"] > MAX_PRODUCTION_FP64_DOTS and feedback:
        raise AdmissionError("feedback exceeded 64 FP64 dots")
    if not skip_gpu:
        dump_json(FIXTURE, payload)
        write_report(payload)
        dump_json(destination / f"{phase}.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    skip = bool(args.skip_gpu or os.environ.get("QW38_OPT074_SKIP_GPU"))
    try:
        result = run_phase(args.phase, args.run_dir, skip_gpu=skip)
    except AdmissionError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
