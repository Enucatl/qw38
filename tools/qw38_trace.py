"""Versioned Quartz diagnostic trace bundles and numeric comparisons.

The v1 container is intentionally narrow: one JSON manifest and one contiguous
little-endian FP32 blob.  Keeping the contract small makes independent readers
easy to write and malformed evidence easy to reject.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "qw38.trace"
VERSION = 1
BLOB_NAME = "tensors.f32le.bin"
DTYPE = "f32-le"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TraceError(ValueError):
    """A trace does not satisfy the pinned v1 contract."""


@dataclass(frozen=True)
class ArtifactIdentity:
    """Pinned name, source revision, and content hash for a trace participant."""

    name: str
    revision: str
    sha256: str


@dataclass(frozen=True)
class TraceTensor:
    """One named tensor captured at a visible execution boundary."""

    name: str
    shape: tuple[int, ...]
    values: Sequence[float]
    role: str
    layer: int | None = None


@dataclass(frozen=True)
class SessionSnapshot:
    """Committed token frontier and checksums of named persistent state."""

    frontier: int
    state_sha256: Mapping[str, str]


@dataclass(frozen=True)
class LoadedTrace:
    """A validated manifest and the exact FP32 values read from its blob."""

    manifest: Mapping[str, Any]
    tensors: Mapping[str, tuple[float, ...]]


@dataclass(frozen=True)
class NonFiniteCounts:
    nan: int
    positive_infinity: int
    negative_infinity: int


@dataclass(frozen=True)
class TopLogitComparison:
    k: int
    expected_token_ids: tuple[int, ...]
    actual_token_ids: tuple[int, ...]
    common_token_count: int
    exact_order: bool
    largest_union_delta: float
    largest_union_delta_token_id: int


@dataclass(frozen=True)
class ComparisonMetrics:
    count: int
    expected_non_finite: NonFiniteCounts
    actual_non_finite: NonFiniteCounts
    maximum_absolute_error: float
    maximum_absolute_error_index: int | None
    maximum_relative_error: float
    maximum_relative_error_index: int | None
    root_mean_square_error: float
    cosine_similarity: float | None
    first_failing_index: int | None
    passed: bool
    top_logits: TopLogitComparison | None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TraceError(f"{label} must be a lowercase SHA-256")
    return value


def _product(shape: Sequence[int]) -> int:
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise TraceError("tensor shapes must contain positive integer dimensions")
    return math.prod(shape)


def _summary(values: Sequence[float]) -> dict[str, int | float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nan_count": sum(math.isnan(value) for value in values),
        "positive_infinity_count": sum(value == math.inf for value in values),
        "negative_infinity_count": sum(value == -math.inf for value in values),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": math.fsum(finite) / len(finite) if finite else None,
        "root_mean_square": (
            math.sqrt(math.fsum(value * value for value in finite) / len(finite))
            if finite
            else None
        ),
    }


def _pack_f32(values: Sequence[float]) -> tuple[bytes, tuple[float, ...]]:
    try:
        raw = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise TraceError(f"value cannot be represented as FP32: {error}") from error
    rounded = struct.unpack(f"<{len(values)}f", raw)
    return raw, rounded


def _identity(identity: ArtifactIdentity, label: str) -> dict[str, str]:
    if not identity.name or not identity.revision:
        raise TraceError(f"{label} identity fields must be non-empty")
    return {
        "name": identity.name,
        "revision": identity.revision,
        "sha256": _checked_sha256(identity.sha256, label),
    }


def _snapshot(snapshot: SessionSnapshot, label: str) -> dict[str, Any]:
    if snapshot.frontier < 0:
        raise TraceError(f"{label} frontier must be non-negative")
    states: dict[str, str] = {}
    for name, digest in sorted(snapshot.state_sha256.items()):
        if not name or name in states:
            raise TraceError(f"{label} state names must be unique and non-empty")
        states[name] = _checked_sha256(digest, f"{label} state {name}")
    return {"frontier": snapshot.frontier, "state_sha256": states}


def write_trace_bundle(
    directory: Path,
    *,
    model: ArtifactIdentity,
    tool: ArtifactIdentity,
    prompt: bytes,
    token_ids: Sequence[int],
    positions: Sequence[int],
    session_before: SessionSnapshot,
    session_after: SessionSnapshot,
    tensors: Sequence[TraceTensor],
) -> Path:
    """Write a deterministic v1 trace and return its manifest path."""

    if len(token_ids) != len(positions):
        raise TraceError("token IDs and positions must have equal lengths")
    if any(not isinstance(value, int) or value < 0 for value in token_ids):
        raise TraceError("token IDs must be non-negative integers")
    if any(not isinstance(value, int) or value < 0 for value in positions):
        raise TraceError("positions must be non-negative integers")
    model_record = _identity(model, "model")
    tool_record = _identity(tool, "tool")
    before = _snapshot(session_before, "session before")
    after = _snapshot(session_after, "session after")

    if directory.exists():
        if not directory.is_dir() or any(directory.iterdir()):
            raise TraceError("trace destination must be an absent or empty directory")
    directory.mkdir(parents=True, exist_ok=True)

    blob = bytearray()
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for tensor in tensors:
        if not tensor.name or tensor.name in names:
            raise TraceError("tensor names must be unique and non-empty")
        if not tensor.role:
            raise TraceError(f"tensor {tensor.name} must have a role")
        if tensor.layer is not None and tensor.layer < 0:
            raise TraceError(f"tensor {tensor.name} layer must be non-negative")
        if _product(tensor.shape) != len(tensor.values):
            raise TraceError(f"tensor {tensor.name} shape does not match its values")
        raw, rounded = _pack_f32(tensor.values)
        offset = len(blob)
        blob.extend(raw)
        records.append(
            {
                "name": tensor.name,
                "role": tensor.role,
                "layer": tensor.layer,
                "shape": list(tensor.shape),
                "dtype": DTYPE,
                "offset_bytes": offset,
                "length_bytes": len(raw),
                "sha256": _sha256(raw),
                "summary": _summary(rounded),
            }
        )
        names.add(tensor.name)

    prompt_record = {
        "encoding": "base64",
        "bytes": base64.b64encode(prompt).decode("ascii"),
        "byte_count": len(prompt),
        "sha256": _sha256(prompt),
        "token_ids": list(token_ids),
        "positions": list(positions),
    }
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "byte_order": "little",
        "model": model_record,
        "tool": tool_record,
        "prompt": prompt_record,
        "session": {"before": before, "after": after},
        "blob": {
            "file": BLOB_NAME,
            "byte_count": len(blob),
            "sha256": _sha256(blob),
        },
        "tensors": records,
    }
    blob_path = directory / BLOB_NAME
    manifest_path = directory / "manifest.json"
    blob_path.write_bytes(blob)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TraceError(f"{label} fields do not match trace v1")
    return value


def read_trace_bundle(directory: Path) -> LoadedTrace:
    """Read and fully validate a v1 trace before returning any tensor values."""

    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceError(f"cannot read trace manifest: {error}") from error
    top = _exact_keys(
        manifest,
        {
            "schema",
            "version",
            "byte_order",
            "model",
            "tool",
            "prompt",
            "session",
            "blob",
            "tensors",
        },
        "manifest",
    )
    if top["schema"] != SCHEMA or top["version"] != VERSION:
        raise TraceError("unsupported trace schema or version")
    if top["byte_order"] != "little":
        raise TraceError("trace v1 requires little-endian storage")
    for label in ("model", "tool"):
        identity = _exact_keys(top[label], {"name", "revision", "sha256"}, label)
        if not isinstance(identity["name"], str) or not identity["name"]:
            raise TraceError(f"{label} name is invalid")
        if not isinstance(identity["revision"], str) or not identity["revision"]:
            raise TraceError(f"{label} revision is invalid")
        _checked_sha256(identity["sha256"], label)

    prompt = _exact_keys(
        top["prompt"],
        {"encoding", "bytes", "byte_count", "sha256", "token_ids", "positions"},
        "prompt",
    )
    if prompt["encoding"] != "base64":
        raise TraceError("trace v1 prompt encoding must be base64")
    try:
        prompt_bytes = base64.b64decode(prompt["bytes"], validate=True)
    except (TypeError, ValueError) as error:
        raise TraceError("prompt is not canonical base64") from error
    if base64.b64encode(prompt_bytes).decode("ascii") != prompt["bytes"]:
        raise TraceError("prompt is not canonical base64")
    if prompt["byte_count"] != len(prompt_bytes):
        raise TraceError("prompt byte count does not match")
    if _checked_sha256(prompt["sha256"], "prompt") != _sha256(prompt_bytes):
        raise TraceError("prompt checksum does not match")
    token_ids = prompt["token_ids"]
    positions = prompt["positions"]
    if not isinstance(token_ids, list) or not isinstance(positions, list):
        raise TraceError("token IDs and positions must be arrays")
    if len(token_ids) != len(positions):
        raise TraceError("token IDs and positions must have equal lengths")
    if any(not isinstance(value, int) or value < 0 for value in token_ids + positions):
        raise TraceError("token IDs and positions must be non-negative integers")

    session = _exact_keys(top["session"], {"before", "after"}, "session")
    for phase in ("before", "after"):
        snapshot = _exact_keys(
            session[phase], {"frontier", "state_sha256"}, f"session {phase}"
        )
        if not isinstance(snapshot["frontier"], int) or snapshot["frontier"] < 0:
            raise TraceError(f"session {phase} frontier is invalid")
        if not isinstance(snapshot["state_sha256"], dict):
            raise TraceError(f"session {phase} state checksums are invalid")
        for name, digest in snapshot["state_sha256"].items():
            if not isinstance(name, str) or not name:
                raise TraceError(f"session {phase} state name is invalid")
            _checked_sha256(digest, f"session {phase} state {name}")

    blob_record = _exact_keys(top["blob"], {"file", "byte_count", "sha256"}, "blob")
    if blob_record["file"] != BLOB_NAME:
        raise TraceError("trace v1 blob name is invalid")
    try:
        raw_blob = (directory / BLOB_NAME).read_bytes()
    except OSError as error:
        raise TraceError(f"cannot read trace blob: {error}") from error
    if blob_record["byte_count"] != len(raw_blob):
        raise TraceError("blob byte count does not match")
    if _checked_sha256(blob_record["sha256"], "blob") != _sha256(raw_blob):
        raise TraceError("blob checksum does not match")

    if not isinstance(top["tensors"], list):
        raise TraceError("tensors must be an array")
    loaded: dict[str, tuple[float, ...]] = {}
    next_offset = 0
    tensor_keys = {
        "name",
        "role",
        "layer",
        "shape",
        "dtype",
        "offset_bytes",
        "length_bytes",
        "sha256",
        "summary",
    }
    for index, item in enumerate(top["tensors"]):
        record = _exact_keys(item, tensor_keys, f"tensor {index}")
        name = record["name"]
        if not isinstance(name, str) or not name or name in loaded:
            raise TraceError("tensor names must be unique and non-empty")
        if not isinstance(record["role"], str) or not record["role"]:
            raise TraceError(f"tensor {name} role is invalid")
        if record["layer"] is not None and (
            not isinstance(record["layer"], int) or record["layer"] < 0
        ):
            raise TraceError(f"tensor {name} layer is invalid")
        shape = record["shape"]
        if not isinstance(shape, list):
            raise TraceError(f"tensor {name} shape is invalid")
        count = _product(shape)
        if record["dtype"] != DTYPE:
            raise TraceError(f"tensor {name} dtype is invalid")
        if record["offset_bytes"] != next_offset:
            raise TraceError(f"tensor {name} is not contiguous")
        expected_length = count * 4
        if record["length_bytes"] != expected_length:
            raise TraceError(f"tensor {name} byte length is invalid")
        end = next_offset + expected_length
        if end > len(raw_blob):
            raise TraceError(f"tensor {name} exceeds the blob")
        raw = raw_blob[next_offset:end]
        if _checked_sha256(record["sha256"], f"tensor {name}") != _sha256(raw):
            raise TraceError(f"tensor {name} checksum does not match")
        values = struct.unpack(f"<{count}f", raw)
        if record["summary"] != _summary(values):
            raise TraceError(f"tensor {name} summary does not match")
        loaded[name] = values
        next_offset = end
    if next_offset != len(raw_blob):
        raise TraceError("blob has unclaimed trailing bytes")
    return LoadedTrace(manifest=top, tensors=loaded)


def _non_finite_counts(values: Sequence[float]) -> NonFiniteCounts:
    return NonFiniteCounts(
        nan=sum(math.isnan(value) for value in values),
        positive_infinity=sum(value == math.inf for value in values),
        negative_infinity=sum(value == -math.inf for value in values),
    )


def compare_values(
    expected: Sequence[float],
    actual: Sequence[float],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    relative_floor: float = 1e-12,
    top_k_logits: int | None = None,
) -> ComparisonMetrics:
    """Compare equal-length arrays using the frozen v1 admission rule.

    A finite element fails only when both its absolute and relative errors exceed
    their tolerances. Every NaN or infinity fails, even if both sides contain the
    same non-finite value. Cosine is intentionally unavailable when any input is
    non-finite or either vector has zero length.
    """

    if len(expected) != len(actual):
        raise TraceError("comparison arrays must have equal lengths")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or not math.isfinite(relative_floor)
        or absolute_tolerance < 0
        or relative_tolerance < 0
        or relative_floor <= 0
    ):
        raise TraceError("comparison tolerances are invalid")
    if top_k_logits is not None and (top_k_logits <= 0 or not expected):
        raise TraceError("top-k logits requires a positive k and non-empty arrays")

    maximum_absolute = 0.0
    maximum_absolute_index: int | None = None
    maximum_relative = 0.0
    maximum_relative_index: int | None = None
    squared_errors: list[float] = []
    first_failure: int | None = None
    all_finite = True
    for index, (reference, candidate) in enumerate(zip(expected, actual, strict=True)):
        if not math.isfinite(reference) or not math.isfinite(candidate):
            all_finite = False
            if first_failure is None:
                first_failure = index
            continue
        absolute = abs(candidate - reference)
        relative = absolute / max(abs(reference), relative_floor)
        squared_errors.append(absolute * absolute)
        if maximum_absolute_index is None or absolute > maximum_absolute:
            maximum_absolute = absolute
            maximum_absolute_index = index
        if maximum_relative_index is None or relative > maximum_relative:
            maximum_relative = relative
            maximum_relative_index = index
        if (
            absolute > absolute_tolerance
            and relative > relative_tolerance
            and first_failure is None
        ):
            first_failure = index

    rms = (
        math.sqrt(math.fsum(squared_errors) / len(squared_errors))
        if squared_errors
        else 0.0
    )
    cosine: float | None = None
    if all_finite and expected:
        reference_norm = math.sqrt(math.fsum(value * value for value in expected))
        candidate_norm = math.sqrt(math.fsum(value * value for value in actual))
        if reference_norm != 0.0 and candidate_norm != 0.0:
            cosine = math.fsum(
                reference * candidate
                for reference, candidate in zip(expected, actual, strict=True)
            ) / (reference_norm * candidate_norm)

    top: TopLogitComparison | None = None
    if top_k_logits is not None:
        if not all(
            math.isfinite(value) for values in (expected, actual) for value in values
        ):
            raise TraceError("top-logit comparison requires finite values")
        k = min(top_k_logits, len(expected))
        expected_ids = tuple(
            sorted(range(len(expected)), key=lambda i: (-expected[i], i))[:k]
        )
        actual_ids = tuple(
            sorted(range(len(actual)), key=lambda i: (-actual[i], i))[:k]
        )
        union = set(expected_ids) | set(actual_ids)
        delta_token = min(union)
        delta = abs(actual[delta_token] - expected[delta_token])
        for token_id in sorted(union):
            candidate_delta = abs(actual[token_id] - expected[token_id])
            if candidate_delta > delta:
                delta = candidate_delta
                delta_token = token_id
        top = TopLogitComparison(
            k=k,
            expected_token_ids=expected_ids,
            actual_token_ids=actual_ids,
            common_token_count=len(set(expected_ids) & set(actual_ids)),
            exact_order=expected_ids == actual_ids,
            largest_union_delta=delta,
            largest_union_delta_token_id=delta_token,
        )

    return ComparisonMetrics(
        count=len(expected),
        expected_non_finite=_non_finite_counts(expected),
        actual_non_finite=_non_finite_counts(actual),
        maximum_absolute_error=maximum_absolute,
        maximum_absolute_error_index=maximum_absolute_index,
        maximum_relative_error=maximum_relative,
        maximum_relative_error_index=maximum_relative_index,
        root_mean_square_error=rms,
        cosine_similarity=cosine,
        first_failing_index=first_failure,
        passed=first_failure is None,
        top_logits=top,
    )
