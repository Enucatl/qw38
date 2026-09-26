"""Checks for the single-run PERF-01 protocol and timing boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from task027_benchmark import llama_settings, observed_ratio, read_run


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
