from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from tools.qw38_trace import (
    BLOB_NAME,
    ArtifactIdentity,
    SessionSnapshot,
    TraceError,
    TraceTensor,
    compare_values,
    read_trace_bundle,
    write_trace_bundle,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def write_example(directory: Path) -> Path:
    return write_trace_bundle(
        directory,
        model=ArtifactIdentity(
            "Qwen3.8-27B-Q4_K_M.gguf", "pinned-model", digest("model")
        ),
        tool=ArtifactIdentity("qw38-eval", "abc123", digest("tool")),
        prompt="café\n".encode(),
        token_ids=[11, 22],
        positions=[7, 8],
        session_before=SessionSnapshot(
            frontier=7, state_sha256={"gdn.0": digest("before")}
        ),
        session_after=SessionSnapshot(
            frontier=9, state_sha256={"gdn.0": digest("after")}
        ),
        tensors=[
            TraceTensor(
                name="layer.0.input_norm",
                role="input_norm",
                layer=0,
                shape=(2,),
                values=[1.0, -2.5],
            ),
            TraceTensor(
                name="logits",
                role="logits",
                shape=(2, 2),
                values=[0.1, float("nan"), float("inf"), float("-inf")],
            ),
        ],
    )


def rewrite_manifest(directory: Path, mutate: object) -> None:
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    assert callable(mutate)
    mutate(document)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def test_trace_round_trip_has_exact_little_endian_layout(tmp_path: Path) -> None:
    manifest_path = write_example(tmp_path / "trace")
    loaded = read_trace_bundle(tmp_path / "trace")

    assert manifest_path.name == "manifest.json"
    assert (tmp_path / "trace" / BLOB_NAME).read_bytes()[:8] == struct.pack(
        "<2f", 1.0, -2.5
    )
    assert loaded.tensors["layer.0.input_norm"] == (1.0, -2.5)
    logits = loaded.tensors["logits"]
    assert logits[0] == pytest.approx(struct.unpack("<f", struct.pack("<f", 0.1))[0])
    assert math.isnan(logits[1])
    assert logits[2:] == (math.inf, -math.inf)

    manifest = loaded.manifest
    assert manifest["schema"] == "qw38.trace"
    assert manifest["version"] == 1
    assert manifest["prompt"]["token_ids"] == [11, 22]
    assert manifest["prompt"]["positions"] == [7, 8]
    assert manifest["session"]["before"]["frontier"] == 7
    assert manifest["session"]["after"]["frontier"] == 9
    assert manifest["tensors"][1]["summary"] == {
        "count": 4,
        "finite_count": 1,
        "maximum": pytest.approx(0.1),
        "mean": pytest.approx(0.1),
        "minimum": pytest.approx(0.1),
        "nan_count": 1,
        "negative_infinity_count": 1,
        "positive_infinity_count": 1,
        "root_mean_square": pytest.approx(0.1),
    }


def test_trace_output_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_example(first)
    write_example(second)

    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / BLOB_NAME).read_bytes() == (second / BLOB_NAME).read_bytes()
    with pytest.raises(TraceError, match="absent or empty"):
        write_example(first)


def test_trace_writer_rejects_invalid_structure(tmp_path: Path) -> None:
    common = {
        "model": ArtifactIdentity("model", "revision", digest("model")),
        "tool": ArtifactIdentity("tool", "revision", digest("tool")),
        "prompt": b"x",
        "token_ids": [1],
        "positions": [0],
        "session_before": SessionSnapshot(0, {}),
        "session_after": SessionSnapshot(1, {}),
    }
    with pytest.raises(TraceError, match="shape"):
        write_trace_bundle(
            tmp_path / "bad-shape",
            **common,
            tensors=[TraceTensor("x", (2,), [1.0], "tap")],
        )
    with pytest.raises(TraceError, match="equal lengths"):
        write_trace_bundle(
            tmp_path / "bad-tokens",
            **{**common, "positions": []},
            tensors=[],
        )
    with pytest.raises(TraceError, match="SHA-256"):
        write_trace_bundle(
            tmp_path / "bad-state",
            **{
                **common,
                "session_after": SessionSnapshot(1, {"gdn.0": "not-a-hash"}),
            },
            tensors=[],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["blob"].update(byte_count=999), "byte count"),
        (lambda document: document["prompt"].update(token_ids=[]), "equal lengths"),
        (lambda document: document["tensors"][0].update(dtype="f32"), "dtype"),
        (lambda document: document["tensors"][1].update(offset_bytes=4), "contiguous"),
        (lambda document: document["tensors"][0].update(shape=[3]), "byte length"),
        (lambda document: document.update(version=2), "unsupported"),
    ],
)
def test_trace_reader_rejects_malformed_manifest(
    tmp_path: Path, mutation: object, message: str
) -> None:
    directory = tmp_path / "trace"
    write_example(directory)
    rewrite_manifest(directory, mutation)

    with pytest.raises(TraceError, match=message):
        read_trace_bundle(directory)


def test_trace_reader_detects_blob_corruption(tmp_path: Path) -> None:
    directory = tmp_path / "trace"
    write_example(directory)
    path = directory / BLOB_NAME
    damaged = bytearray(path.read_bytes())
    damaged[0] ^= 1
    path.write_bytes(damaged)

    with pytest.raises(TraceError, match="blob checksum"):
        read_trace_bundle(directory)


def test_comparison_reports_each_numeric_metric() -> None:
    metrics = compare_values(
        [1.0, -2.0, 0.0],
        [1.1, -2.0, 0.001],
        absolute_tolerance=0.01,
        relative_tolerance=0.05,
        relative_floor=0.01,
    )

    assert metrics.maximum_absolute_error == pytest.approx(0.1)
    assert metrics.maximum_absolute_error_index == 0
    assert metrics.maximum_relative_error == pytest.approx(0.1)
    assert metrics.maximum_relative_error_index == 0
    assert metrics.root_mean_square_error == pytest.approx(
        math.sqrt((0.1**2 + 0.001**2) / 3)
    )
    assert metrics.cosine_similarity == pytest.approx(0.9992318548359757)
    assert metrics.first_failing_index == 0
    assert not metrics.passed


def test_comparison_uses_and_rule_for_finite_tolerances() -> None:
    absolute_pass = compare_values(
        [1000.0],
        [1000.1],
        absolute_tolerance=0.01,
        relative_tolerance=0.001,
    )
    relative_pass = compare_values(
        [0.0],
        [0.001],
        absolute_tolerance=0.01,
        relative_tolerance=0.0,
    )

    assert absolute_pass.passed
    assert relative_pass.passed


def test_comparison_reports_non_finite_values_and_first_mismatch() -> None:
    metrics = compare_values(
        [float("nan"), math.inf, -math.inf, 4.0],
        [float("nan"), math.inf, math.inf, 4.0],
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )

    assert metrics.expected_non_finite.nan == 1
    assert metrics.expected_non_finite.positive_infinity == 1
    assert metrics.expected_non_finite.negative_infinity == 1
    assert metrics.actual_non_finite.positive_infinity == 2
    assert metrics.first_failing_index == 0
    assert metrics.cosine_similarity is None
    assert not metrics.passed


def test_comparison_reports_top_logit_order_and_near_tie() -> None:
    metrics = compare_values(
        [0.0, 5.0, 4.999, -1.0],
        [0.0, 4.998, 5.001, -1.0],
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        top_k_logits=2,
    )

    assert metrics.passed
    assert metrics.top_logits is not None
    assert metrics.top_logits.expected_token_ids == (1, 2)
    assert metrics.top_logits.actual_token_ids == (2, 1)
    assert metrics.top_logits.common_token_count == 2
    assert not metrics.top_logits.exact_order
    assert metrics.top_logits.largest_union_delta == pytest.approx(0.002)
    assert metrics.top_logits.largest_union_delta_token_id == 2


def test_comparison_rejects_ambiguous_inputs() -> None:
    with pytest.raises(TraceError, match="equal lengths"):
        compare_values([1.0], [], absolute_tolerance=0.0, relative_tolerance=0.0)
    with pytest.raises(TraceError, match="finite"):
        compare_values(
            [math.inf],
            [math.inf],
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            top_k_logits=1,
        )
