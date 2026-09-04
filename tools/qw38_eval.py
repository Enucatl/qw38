"""Typed client and strict reader for the qw38-eval harness."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tools.qw38_trace import LoadedTrace, TraceError, read_trace_bundle

VOCABULARY_SIZE = 248_320
TOKEN_LIMIT = VOCABULARY_SIZE
TRACE_FILTERS = frozenset(
    {
        "0:layer_residual",
        "3:layer_residual",
        "63:layer_residual",
        "global:final_norm",
        "global:logits",
    }
)


class EvalError(ValueError):
    """Malformed request or evidence."""


class EvalProcessError(RuntimeError):
    """The native harness failed to execute successfully."""

    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        super().__init__(
            f"qw38-eval exited with status {result.returncode}: {result.stderr}"
        )
        self.result = result


def parse_tokens(value: str, *, label: str = "tokens") -> tuple[int, ...]:
    if not value or any(ch.isspace() for ch in value):
        raise EvalError(f"{label} must be a non-empty comma-separated list")
    fields = value.split(",")
    if any(
        not field or not field.isascii() or not field.isdecimal() for field in fields
    ):
        raise EvalError(f"{label} contains an invalid token")
    tokens = tuple(int(field, 10) for field in fields)
    if any(token >= TOKEN_LIMIT for token in tokens):
        raise EvalError(f"{label} contains an out-of-vocabulary token")
    return tokens


@dataclass(frozen=True)
class EvalRequest:
    model: Path
    tokens: tuple[int, ...]
    output: Path
    source_revision: str
    source_state: str

    def __post_init__(self) -> None:
        if not self.tokens:
            raise EvalError("tokens must be non-empty")
        if not self.source_revision or self.source_state not in {"clean", "dirty"}:
            raise EvalError("source identity is invalid")


@dataclass(frozen=True)
class LogitsRequest(EvalRequest):
    pass


@dataclass(frozen=True)
class CheckpointRequest(EvalRequest):
    continuation: tuple[int, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.continuation:
            raise EvalError("continuation must be non-empty")


@dataclass(frozen=True)
class TraceRequest(EvalRequest):
    trace_filters: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.trace_filters or len(set(self.trace_filters)) != len(
            self.trace_filters
        ):
            raise EvalError("trace filters must be non-empty and distinct")
        if any(value not in TRACE_FILTERS for value in self.trace_filters):
            raise EvalError("trace filter is not pinned")


@dataclass(frozen=True)
class LogitsResult:
    record: dict[str, object]
    logits: tuple[float, ...]


@dataclass(frozen=True)
class CheckpointResult:
    """Authenticated checkpoint continuation evidence."""

    record: dict[str, object]
    logits: tuple[float, ...]


@dataclass(frozen=True)
class TraceResult:
    """Validated qw38.trace v1 evidence returned by a diagnostic run."""

    record: dict[str, object]
    manifest: LoadedTrace


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EvalError(f"{label} has unexpected fields")
    return value


def _model_identity(value: object) -> dict[str, object]:
    result = _exact(value, {"name", "revision", "sha256", "byte_count"}, "model")
    if not isinstance(result["name"], str) or not result["name"]:
        raise EvalError("model name is invalid")
    if not isinstance(result["revision"], str) or not result["revision"]:
        raise EvalError("model revision is invalid")
    if not isinstance(result["sha256"], str) or len(result["sha256"]) != 64:
        raise EvalError("model SHA-256 is invalid")
    if not isinstance(result["byte_count"], int) or result["byte_count"] < 0:
        raise EvalError("model byte count is invalid")
    if not isinstance(result["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", result["sha256"]
    ):
        raise EvalError("model SHA-256 is invalid")
    return result


def _tool_identity(value: object) -> dict[str, object]:
    result = _exact(value, {"name", "revision", "source_state", "sha256"}, "tool")
    if not isinstance(result["name"], str) or not result["name"]:
        raise EvalError("tool name is invalid")
    if not isinstance(result["revision"], str) or not result["revision"]:
        raise EvalError("tool revision is invalid")
    if result["source_state"] not in {"clean", "dirty"}:
        raise EvalError("tool source state is invalid")
    if not isinstance(result["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", result["sha256"]
    ):
        raise EvalError("tool SHA-256 is invalid")
    return result


def _runtime(value: object) -> dict[str, object]:
    result = _exact(
        value,
        {
            "backend",
            "cuda_target",
            "cuda_runtime_version",
            "cuda_driver_version",
            "device_name",
            "compute_capability",
        },
        "runtime",
    )
    if result["backend"] not in {"cuda", "host"}:
        raise EvalError("runtime backend is invalid")
    return result


def _summary(value: object, values: tuple[float, ...], label: str) -> None:
    result = _exact(
        value,
        {
            "count",
            "finite_count",
            "nan_count",
            "positive_infinity_count",
            "negative_infinity_count",
            "minimum",
            "maximum",
            "mean",
            "root_mean_square",
        },
        label,
    )
    if result["count"] != len(values):
        raise EvalError(f"{label} count does not match blob")
    finite = tuple(v for v in values if math.isfinite(v))
    expected = {
        "finite_count": len(finite),
        "nan_count": sum(math.isnan(v) for v in values),
        "positive_infinity_count": sum(v == math.inf for v in values),
        "negative_infinity_count": sum(v == -math.inf for v in values),
    }
    if any(result[key] != val for key, val in expected.items()):
        raise EvalError(f"{label} non-finite counts do not match blob")
    if finite:
        mean = math.fsum(finite) / len(finite)
        rms = math.sqrt(math.fsum(v * v for v in finite) / len(finite))
        if result["minimum"] != min(finite) or result["maximum"] != max(finite):
            raise EvalError(f"{label} extrema do not match blob")
        for key, expected_value in (("mean", mean), ("root_mean_square", rms)):
            if not isinstance(result[key], (int, float)) or not math.isclose(
                float(result[key]), expected_value, rel_tol=2e-6, abs_tol=2e-6
            ):
                raise EvalError(f"{label} {key} does not match blob")


def read_logits_result(directory: Path) -> LogitsResult:
    """Validate a logits result and return immutable complete FP32 values."""
    try:
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        blob = (directory / "logits.f32le.bin").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise EvalError(f"cannot read logits result: {error}") from error
    top = _exact(
        record,
        {
            "schema",
            "version",
            "mode",
            "status",
            "model",
            "tool",
            "runtime",
            "tokens",
            "positions",
            "frontier",
            "logits",
            "greedy_token",
        },
        "result",
    )
    if (
        top["schema"] != "qw38.eval-result"
        or top["version"] != 1
        or top["mode"] != "logits"
        or top["status"] != "ok"
    ):
        raise EvalError("unsupported logits result")
    _model_identity(top["model"])
    _tool_identity(top["tool"])
    _runtime(top["runtime"])
    tokens = top["tokens"]
    positions = top["positions"]
    if (
        not isinstance(tokens, list)
        or not isinstance(positions, list)
        or len(tokens) != len(positions)
    ):
        raise EvalError("token metadata is invalid")
    parse_tokens(",".join(str(token) for token in tokens))
    if positions != list(range(len(tokens))) or top["frontier"] != len(tokens):
        raise EvalError("token frontier metadata is invalid")
    if len(blob) != VOCABULARY_SIZE * 4:
        raise EvalError("logits blob has invalid byte count")
    logits = struct.unpack(f"<{VOCABULARY_SIZE}f", blob)
    if any(not math.isfinite(value) for value in logits):
        raise EvalError("logits must be finite")
    summary = _exact(
        top["logits"],
        {"file", "dtype", "shape", "byte_count", "sha256", "summary"},
        "logits",
    )
    if (
        summary["file"] != "logits.f32le.bin"
        or summary["dtype"] != "f32-le"
        or summary["shape"] != [VOCABULARY_SIZE]
        or summary["byte_count"] != len(blob)
        or summary["sha256"] != _sha256(blob)
    ):
        raise EvalError("logits identity does not match blob")
    _summary(summary["summary"], logits, "logits")
    greedy = min(range(len(logits)), key=lambda i: (-logits[i], i))
    if top["greedy_token"] != greedy:
        raise EvalError("greedy token does not match logits")
    return LogitsResult(record=top, logits=logits)


def read_checkpoint_result(directory: Path) -> CheckpointResult:
    """Validate checkpoint metadata, checkpoint bytes, and continuation logits."""
    try:
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        checkpoint = (directory / "checkpoint.qw38").read_bytes()
        blob = (directory / "continuation_logits.f32le.bin").read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise EvalError(f"cannot read checkpoint result: {error}") from error
    top = _exact(
        record,
        {
            "schema",
            "version",
            "mode",
            "status",
            "model",
            "tool",
            "runtime",
            "prefix_tokens",
            "continuation_tokens",
            "prefix_positions",
            "continuation_positions",
            "frontiers",
            "checkpoint",
            "continuation_logits",
            "greedy_token",
            "equality",
        },
        "checkpoint result",
    )
    if (
        top["schema"] != "qw38.eval-result"
        or top["version"] != 1
        or top["mode"] != "checkpoint"
        or top["status"] != "ok"
    ):
        raise EvalError("unsupported checkpoint result")
    _model_identity(top["model"])
    _tool_identity(top["tool"])
    _runtime(top["runtime"])
    for label in ("prefix_tokens", "continuation_tokens"):
        values = top[label]
        if not isinstance(values, list):
            raise EvalError(f"{label} is invalid")
        parse_tokens(",".join(str(value) for value in values), label=label)
    prefix = top["prefix_tokens"]
    continuation = top["continuation_tokens"]
    if top["prefix_positions"] != list(range(len(prefix))) or top[
        "continuation_positions"
    ] != list(range(len(prefix), len(prefix) + len(continuation))):
        raise EvalError("checkpoint positions are invalid")
    frontiers = _exact(
        top["frontiers"],
        {"prefix", "uninterrupted_final", "restored_prefix", "restored_final"},
        "frontiers",
    )
    if (
        frontiers["prefix"] != len(prefix)
        or frontiers["uninterrupted_final"] != len(prefix) + len(continuation)
        or frontiers["restored_prefix"] != len(prefix)
        or frontiers["restored_final"] != len(prefix) + len(continuation)
    ):
        raise EvalError("checkpoint frontiers are invalid")
    checkpoint_summary = _exact(
        top["checkpoint"], {"file", "byte_count", "sha256"}, "checkpoint"
    )
    if checkpoint_summary["file"] != "checkpoint.qw38":
        raise EvalError("checkpoint file name is invalid")
    if checkpoint_summary["byte_count"] != len(checkpoint) or checkpoint_summary[
        "sha256"
    ] != _sha256(checkpoint):
        raise EvalError("checkpoint identity does not match bytes")
    equality = _exact(top["equality"], {"tokens", "logits"}, "equality")
    if equality["tokens"] is not True or equality["logits"] is not True:
        raise EvalError("checkpoint equality was not proven")
    logits_record = _exact(
        top["continuation_logits"],
        {"file", "dtype", "shape", "byte_count", "sha256", "summary"},
        "continuation logits",
    )
    if (
        logits_record["file"] != "continuation_logits.f32le.bin"
        or logits_record["dtype"] != "f32-le"
        or logits_record["shape"] != [VOCABULARY_SIZE]
        or logits_record["byte_count"] != len(blob)
        or logits_record["sha256"] != _sha256(blob)
    ):
        raise EvalError("checkpoint logits identity does not match blob")
    logits = struct.unpack(f"<{len(blob) // 4}f", blob)
    if any(not math.isfinite(value) for value in logits):
        raise EvalError("checkpoint logits must be finite")
    _summary(logits_record["summary"], logits, "continuation logits")
    greedy = min(range(len(logits)), key=lambda i: (-logits[i], i))
    if top["greedy_token"] != greedy:
        raise EvalError("checkpoint greedy token does not match logits")
    return CheckpointResult(record=top, logits=logits)


def read_trace_result(directory: Path) -> TraceResult:
    """Validate the eval envelope and the frozen trace bundle it names."""
    try:
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalError(f"cannot read trace result: {error}") from error
    top = _exact(
        record,
        {
            "schema",
            "version",
            "mode",
            "status",
            "model",
            "tool",
            "runtime",
            "tokens",
            "positions",
            "frontier",
            "trace",
        },
        "trace result",
    )
    if (top["schema"], top["version"], top["mode"], top["status"]) != (
        "qw38.eval-result",
        1,
        "trace",
        "ok",
    ):
        raise EvalError("unsupported trace result")
    _model_identity(top["model"])
    _tool_identity(top["tool"])
    _runtime(top["runtime"])
    tokens = top["tokens"]
    positions = top["positions"]
    if not isinstance(tokens, list) or not isinstance(positions, list):
        raise EvalError("trace token metadata is invalid")
    parse_tokens(",".join(str(value) for value in tokens))
    if positions != list(range(len(tokens))) or top["frontier"] != len(tokens):
        raise EvalError("trace frontier metadata is invalid")
    trace = _exact(
        top["trace"], {"manifest", "blob", "filters", "tensor_names"}, "trace"
    )
    for key, expected in (("manifest", "manifest.json"), ("blob", "tensors.f32le.bin")):
        item = _exact(trace[key], {"file", "byte_count", "sha256"}, f"trace {key}")
        try:
            data = (directory / expected).read_bytes()
        except OSError as error:
            raise EvalError(f"cannot read trace {key}: {error}") from error
        if (
            item["file"] != expected
            or item["byte_count"] != len(data)
            or item["sha256"] != _sha256(data)
        ):
            raise EvalError(f"trace {key} identity does not match bytes")
    filters = trace["filters"]
    names = trace["tensor_names"]
    if (
        not isinstance(filters, list)
        or not isinstance(names, list)
        or len(filters) != len(names)
        or not filters
        or any(not isinstance(value, str) for value in filters)
        or any(not isinstance(value, str) or not value for value in names)
        or len(set(filters)) != len(filters)
        or len(set(names)) != len(names)
        or any(value not in TRACE_FILTERS for value in filters)
    ):
        raise EvalError("trace filter metadata is invalid")
    try:
        loaded = read_trace_bundle(directory)
    except TraceError as error:
        raise EvalError(f"malformed trace bundle: {error}") from error
    if list(loaded.tensors) != names:
        raise EvalError("trace tensor names do not match bundle")
    return TraceResult(record=top, manifest=loaded)


def run_native(
    request: EvalRequest, *, binary: Path
) -> LogitsResult | CheckpointResult | TraceResult:
    """Run a request without shell interpolation."""
    mode = (
        "logits"
        if isinstance(request, LogitsRequest)
        else "checkpoint"
        if isinstance(request, CheckpointRequest)
        else "trace"
    )
    argv = [
        str(binary),
        str(request.model),
        "--mode",
        mode,
        "--tokens",
        ",".join(map(str, request.tokens)),
        "--output",
        str(request.output),
        "--source-revision",
        request.source_revision,
        "--source-state",
        request.source_state,
    ]
    if isinstance(request, CheckpointRequest):
        argv += ["--continuation", ",".join(map(str, request.continuation))]
    if isinstance(request, TraceRequest):
        for trace_filter in request.trace_filters:
            argv += ["--trace-filter", trace_filter]
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise EvalProcessError(result)
    try:
        if isinstance(request, LogitsRequest):
            return read_logits_result(request.output)
        if isinstance(request, CheckpointRequest):
            return read_checkpoint_result(request.output)
        return read_trace_result(request.output)
    except (EvalError, TraceError) as error:
        raise EvalError(
            f"native process produced malformed evidence: {error}"
        ) from error


__all__ = [
    "CheckpointRequest",
    "EvalError",
    "EvalProcessError",
    "LogitsRequest",
    "LogitsResult",
    "CheckpointResult",
    "TraceRequest",
    "parse_tokens",
    "read_logits_result",
    "read_checkpoint_result",
    "read_trace_result",
    "read_trace_bundle",
    "run_native",
]
