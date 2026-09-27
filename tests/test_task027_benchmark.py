"""Checks for the single-run PERF-01 protocol and timing boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import task027_benchmark as benchmark
import task027_profiles as profiles
from task027_benchmark import llama_settings, observed_ratio, read_development, read_run


def test_observed_ratio_units() -> None:
    """Report one observation's ratio without inventing uncertainty."""
    result = observed_ratio(2000, 1000, 128)
    assert result["ratio"] == 0.5
    assert result["qw38_tokens_per_second"] == 64
    assert result["observed_parity"] == "unmet"
    assert result["runs_per_engine"] == 1
    assert not any(key in result for key in ("median", "p99", "confidence"))
    for value in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            observed_ratio(value, 1, 128)


def test_request_capture_rejects_repetition_and_wrong_work(tmp_path: Path) -> None:
    """A single greedy request has 128 outputs and 127 decode inputs."""
    path = tmp_path / "run.jsonl"
    row = {
        "kind": "sample",
        "row": "request-256",
        "total_ms": 137.0,
        "ttft_ms": 10.0,
        "tail_ms": 127.0,
        "populated_setup_ms": 0,
        "steps_ms": [1.0] * 127,
        "output_ids": [123] * 128,
        "free_bytes": 100,
        "final_position": 383,
        "finite_logits": True,
    }
    records = [
        {"kind": "setup", "capacity_requested": 384, "capacity_allocated": 512},
        row,
        {"kind": "complete", "runs": 1, "warmups": 0},
    ]

    def write() -> None:
        """Write the current small capture fixture."""
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    write()
    assert read_run(path, "request", 256)[1]["final_position"] == 383
    records.insert(2, row)
    write()
    with pytest.raises(ValueError, match="repeated"):
        read_run(path, "request", 256)
    records.pop(2)
    row["steps_ms"] = [1.0] * 128
    write()
    with pytest.raises(ValueError, match="boundary"):
        read_run(path, "request", 256)
    row["steps_ms"] = [1.0] * 127
    row["ttft_ms"] = 20
    write()
    with pytest.raises(ValueError, match="accounting"):
        read_run(path, "request", 256)


def test_offload_and_effective_flash_validation() -> None:
    """Missing offload or effective flash evidence cannot become a baseline."""
    log = (
        "offloaded 65/65 layers to GPU\nresolve_fused_ops: Flash Attention enabled\n"
        "CUDA0 model buffer size = 18084.41 MiB\nCUDA Graph id 16 reused"
    )
    assert llama_settings(log)["buffers_mib"]["CUDA0_model"] == 18084.41
    for bad in (
        log.replace("65/65", "64/65"),
        log + "\nCPU model buffer size = 1 MiB",
        log.replace("Flash Attention enabled", "Flash Attention disabled"),
    ):
        with pytest.raises(ValueError):
            llama_settings(bad)


def test_development_capture_requires_exact_work_and_finite_accounting(
    tmp_path: Path,
) -> None:
    """Reject captures with a wrong workload, missing work or invalid timing."""
    path = tmp_path / "development.jsonl"
    rows = [
        {
            "kind": "setup",
            "capacity_requested": 384,
            "capacity_allocated": 512,
            "profiling_enabled": True,
        },
        {"kind": "diagnostic", "next_fixed_token": 123, "target_nll": 1.0},
        {
            "kind": "sample",
            "row": "development-256",
            "steps_ms": [1.0] * 8,
            "final_position": 264,
            "finite_logits": True,
            "output_ids": [123] * 9,
            "total_ms": 18.0,
            "ttft_ms": 10.0,
            "tail_ms": 8.0,
            "populated_setup_ms": 0.0,
            "free_bytes": 100,
        },
        {"kind": "complete", "runs": 1, "warmups": 0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert read_development(path, 256)["sample"]["final_position"] == 264
    for index, key, value in [
        (0, "capacity_requested", 256),
        (0, "capacity_allocated", 383),
        (0, "profiling_enabled", 1),
        (1, "target_nll", float("nan")),
        (2, "row", "decode-256"),
        (2, "steps_ms", [1.0] * 7),
        (2, "steps_ms", [1.0] * 7 + [float("inf")]),
        (2, "steps_ms", [20.0] * 8),
        (2, "final_position", 263),
        (2, "output_ids", [123] * 8),
        (2, "output_ids", [True] * 9),
        (2, "output_ids", [248320] * 9),
        (2, "finite_logits", False),
        (2, "total_ms", 0),
        (2, "tail_ms", 5.0),
        (3, "runs", 2),
    ]:
        changed = [dict(row) for row in rows]
        changed[index][key] = value
        path.write_text("\n".join(json.dumps(row) for row in changed) + "\n")
        with pytest.raises(ValueError):
            read_development(path, 256)
    path.write_text("\n".join(json.dumps(row) for row in rows + [rows[-1]]) + "\n")
    with pytest.raises(ValueError, match="repeated"):
        read_development(path, 256)


@pytest.mark.parametrize("changed_logits", [False, True])
def test_telemetry_capture_checks_outputs_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_logits: bool
) -> None:
    """Reject profiler-altered logits and persist a failed capture manifest."""
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    binary = tmp_path / benchmark.CANDIDATE
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"candidate")
    tokens = (
        tmp_path / ".cache/evaluation/qw38-language-v2/task027-support/tokens-256.u32le"
    )
    tokens.parent.mkdir(parents=True)
    tokens.write_bytes(b"fixed tokens")
    artifact = tmp_path / "model.qw38"
    artifact.write_bytes(b"manifest-only fixture")
    parser = tmp_path / "scripts/task027_profiles.py"
    parser.parent.mkdir()
    parser.write_text("# parser fixture\n")
    monkeypatch.setattr(benchmark, "thermal", lambda: "test GPU")
    monkeypatch.setattr(
        benchmark,
        "command_output",
        lambda command: "" if command[0] == "nvidia-smi" else "revision",
    )
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *args, **kwargs: ""
    )

    def run(command: list[str], **kwargs: object) -> None:
        """Write the same output files as the request adapter, without a GPU."""
        if "development" not in command:
            return
        output = Path(command[-1])
        profiled = "QW38_PROFILE=1" in command
        records = [
            {
                "kind": "setup",
                "manifest_digest": "identity",
                "capacity_requested": 384,
                "capacity_allocated": 384,
                "profiling_enabled": profiled,
            },
            {"kind": "diagnostic", "target_nll": 1.0, "next_fixed_token": 123},
            {
                "kind": "sample",
                "row": "development-256",
                "steps_ms": [1.0] * 8,
                "final_position": 264,
                "finite_logits": True,
                "output_ids": [123] * 9,
                "total_ms": 18.0,
                "ttft_ms": 10.0,
                "tail_ms": 8.0,
                "populated_setup_ms": 0.0,
                "free_bytes": 100,
            },
            {"kind": "complete", "runs": 1, "warmups": 0},
        ]
        output.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        Path(str(output) + ".logits.f32").write_bytes(
            b"changed" if changed_logits and profiled else b"identical"
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(
        profiles, "telemetry", lambda path: {"unmapped_inference": [], "windows": []}
    )
    directory = tmp_path / "capture"
    if changed_logits:
        with pytest.raises(ValueError, match="profiling changed"):
            benchmark.capture_telemetry(directory, str(artifact), 256, True)
    else:
        benchmark.capture_telemetry(directory, str(artifact), 256, True)
        report = json.loads((directory / "telemetry.json").read_text())
        assert report["instrumentation_comparison"]["outputs_byte_identical"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == ("FAILED" if changed_logits else "COMPLETE")
    assert len(manifest["commands"]) == 3
