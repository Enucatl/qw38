# /// script
# requires-python = "==3.12.*"
# ///
"""Capture matched PERF-01 trials and report every row without averaging losses."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path

from task018_validate_fixtures import sha256_file

ROOT = Path.cwd()
CACHE = Path(".cache/task027")
LLAMA_IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6"
CANDIDATE = "build/pinned-release/benchmarks/qw38_bench_request"
LLAMA = ".cache/task027/llama-request-bench"
ARTIFACT = ".cache/candidates/candidate-v2-q4k-rope-fixed.qw38"
WORKLOADS = [
    (mode, depth)
    for mode, depths in (
        ("decode", (512, 4096, 32768)),
        ("request", (256, 4096, 32768)),
    )
    for depth in depths
]


def command_output(command: list[str]) -> str:
    """Run a bounded metadata command and return its output."""
    return subprocess.check_output(command, text=True).strip()


def thermal() -> str:
    """Record thermal, power and clock conditions at a block boundary."""
    return command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used,temperature.gpu,"
            "power.draw,power.limit,clocks.current.graphics,clocks.current.memory,pstate",
            "--format=csv,noheader",
        ]
    )


def identity() -> dict:
    """Identify code, binaries and metadata, never model tensor payloads."""
    files = [
        CANDIDATE,
        LLAMA,
        "scripts/task027_benchmark.py",
        "scripts/task027_prepare.py",
        "benchmarks/request_bench.cpp",
        "build/pinned-release/src/libqw38_runtime.a",
        "build/pinned-release/cuda/libqw38_cuda.a",
        "build/pinned-release/src/qw38-evaluate",
        "docs/architecture/evaluation-policy-v0.md",
        "docs/architecture/evaluation-policy-core-54.md",
        str(CACHE / "inputs.json"),
    ]
    files += [str(path) for path in sorted((CACHE / "llama-headers").glob("*.h"))]
    files += [str(path) for path in sorted((CACHE / "llama-lib").glob("*.so.0"))]
    inputs = json.loads((CACHE / "inputs.json").read_text())
    if inputs["status"] != "VALID":
        raise ValueError("token identity validation has not passed")
    if (
        sha256_file(Path("build/pinned-release/src/qw38-evaluate"))
        != "ab7479715a658fb9072322f87dba4390aa0f1bcda2a38aeaf6052b0da7577c1c"
    ):
        raise ValueError("TASK-026 accepted evaluator identity changed")
    for item in inputs["inputs"].values():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("frozen token input changed")
        files.append(item["path"])
    artifact = Path(ARTIFACT).stat()
    gguf = Path("models/Qwen3.8-27B-Q4_K_M.gguf").stat()
    return {
        "sha256": {path: sha256_file(Path(path)) for path in files},
        "artifact_stat": {"size": artifact.st_size, "mtime_ns": artifact.st_mtime_ns},
        "gguf_stat": {"size": gguf.st_size, "mtime_ns": gguf.st_mtime_ns},
        "candidate_image": command_output(
            [
                "docker",
                "image",
                "inspect",
                "qw38-dev:cuda13.4.1-pinned",
                "--format",
                "{{.Id}}",
            ]
        ),
        "llama_image": LLAMA_IMAGE,
    }


def read_run(path: Path, mode: str, depth: int) -> tuple[dict, dict]:
    """Validate one execution, its token counts and exact timing boundaries."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 3 or [r["kind"] for r in records] != [
        "setup",
        "sample",
        "complete",
    ]:
        raise ValueError(f"incomplete or repeated capture: {path}")
    setup, row, complete = records
    if complete != {"kind": "complete", "runs": 1, "warmups": 0}:
        raise ValueError("exactly one run and zero warmups required")
    if (
        setup["capacity_requested"] != depth + 128
        or setup["capacity_allocated"] < depth + 128
    ):
        raise ValueError("incorrect context capacity")
    steps = 128 if mode == "decode" else 127
    if (
        row["row"] != f"{mode}-{depth}"
        or len(row["steps_ms"]) != steps
        or row["final_position"] != depth + steps
    ):
        raise ValueError("wrong workload boundary or position")
    if len(row["output_ids"]) != 128 or any(
        not isinstance(token, int) or not 0 <= token < 248320
        for token in row["output_ids"]
    ):
        raise ValueError("invalid output token work")
    if not row["finite_logits"] or any(
        not math.isfinite(value) or value <= 0
        for value in [
            row["total_ms"],
            row["tail_ms"],
            row["free_bytes"],
            *row["steps_ms"],
        ]
    ):
        raise ValueError("nonpositive or nonfinite measurement")
    if mode == "decode":
        if row["ttft_ms"] != 0 or row["populated_setup_ms"] <= 0:
            raise ValueError("decode setup must be outside its timer")
    elif (
        row["populated_setup_ms"] != 0
        or not 0 < row["ttft_ms"] < row["total_ms"]
        or abs(row["total_ms"] - row["ttft_ms"] - row["tail_ms"]) > 1
    ):
        raise ValueError("request timing accounting mismatch")
    return setup, row


def llama_settings(log: str) -> dict:
    """Require full GPU weights and record resolved production scheduling."""
    if "offloaded 65/65 layers to GPU" not in log or re.search(
        r"(?:CPU|CUDA_Host) model buffer size", log
    ):
        raise ValueError("llama language weights are not fully GPU resident")
    if "resolve_fused_ops: Flash Attention enabled" not in log:
        raise ValueError(
            "default flash attention did not resolve to the validated path"
        )
    buffers = re.findall(
        r"(CUDA0|CUDA_Host)\s+(model|KV|RS|compute|output) buffer size\s*=\s*([0-9.]+) MiB",
        log,
    )
    return {
        "flash_attention": "enabled",
        "gpu_layers": "65/65 including output and embedding override",
        "buffers_mib": {
            f"{device}_{kind}": float(size) for device, kind, size in buffers
        },
        "cuda_graph_reuse_observed": "CUDA Graph id" in log and "reused" in log,
    }


def capture() -> None:
    """Collect one profiled run per engine/workload, with no separate reruns."""
    directory = CACHE / "single-run"
    directory.mkdir(exist_ok=False)
    identities = identity()
    nsys_root = "/opt/nvidia/nsight-compute/2026.3.0/host"
    nsys = nsys_root + "/target-linux-x64/nsys"
    manifest = {
        "status": "IN_PROGRESS",
        "identities": identities,
        "source_revision": command_output(["git", "rev-parse", "HEAD"]),
        "llama_source_revision": "e6ab7c1a41054a888ada952eab4c886444c2f5ad",
        "llama_original_inspection_revision": "1945e092030f8668ff93382799502d01490e564d",
        "protocol": "2026-09-26 user amendment: one run, zero warmups, no repetition statistics",
        "quality_context": "TASK-026 accepted runtime and artifact; speed comparison, no new Pareto claim",
        "timing_units": "host monotonic milliseconds; synchronized logits and argmax",
        "instrumentation": "Nsight Systems 2026.3 CUDA/NVTX trace and allocation tracking in the same run; latencies include profiling overhead",
        "prefill_source": "the complete request's TTFT is its prompt ingestion/final-logits/argmax measurement",
        "cold_definition": "fresh process CUDA initialization, model load/upload and session creation reported separately; OS file cache is not evicted",
        "first_use": "lazy workspace and graph setup are included in the single measured execution; decode prompt population is separately timed setup",
        "output_policy": "128 greedy tokens, EOS ignored without suppression; 127 decode tail calls; populated decode uses 128 fixed inputs",
        "generation_vs_evaluation": "final-position logits only; requested-row evaluation excluded",
        "frozen_settings": {
            "qw38": "chunk=256; BF16 activations/KV; FP32 recurrent state; Q4_K/Q8; no graphs",
            "llama": "n_gpu_layers=-1; embedding on GPU; batch=2048; microbatch=512; threads=4; F16 KV; FP32 recurrent state; auto flash resolves enabled; production CUDA graphs",
            "capacity": "requested T+128; llama internally rounds allocation to 256; only the requested logical capacity is used",
            "concurrency": "one GPU process, one fresh sequence; no MTP/speculation or cross-request cache",
            "clocks": "driver-managed clocks; existing 400 W power limit; no tuning",
        },
        "gguf_provenance_limit": "GGUF carries name and architecture but no source/conversion revision; exact source provenance is unavailable",
        "run_order": [],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for index, (mode, depth) in enumerate(WORKLOADS):
        for engine in ("qw38", "llama") if index % 2 == 0 else ("llama", "qw38"):
            active = command_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ]
            )
            if active:
                raise RuntimeError(f"GPU is not idle before benchmark: {active}")
            prefix = directory / f"{mode}-{depth}-{engine}"
            binary = CANDIDATE if engine == "qw38" else LLAMA
            model = ARTIFACT if engine == "qw38" else "models/Qwen3.8-27B-Q4_K_M.gguf"
            image = identities["candidate_image"] if engine == "qw38" else LLAMA_IMAGE
            command = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-u",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{ROOT}:{ROOT}",
                "-v",
                f"{ROOT / CACHE / 'nsight-host'}:{nsys_root}:ro",
                "-w",
                str(ROOT),
                "-e",
                "LD_LIBRARY_PATH=/app",
                "-e",
                "QW38_PROFILE=1",
                "--entrypoint",
                nsys,
                image,
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-memory-usage=true",
                "--cuda-graph-trace=node",
                "--sample=none",
                "--cpuctxsw=none",
                "-o",
                str(prefix),
                str(ROOT / binary),
                model,
                str(CACHE / f"tokens-{depth}.u32le"),
                mode,
                str(depth),
                str(prefix.with_suffix(".jsonl")),
            ]
            record = {
                "engine": engine,
                "row": f"{mode}-{depth}",
                "command": shlex.join(command),
                "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "thermal_before": thermal(),
            }
            print(record["command"], flush=True)
            with prefix.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record.update(
                exit_code=result.returncode,
                thermal_after=thermal(),
                finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            )
            if result.returncode == 0 and engine == "llama":
                record["effective_settings"] = llama_settings(
                    prefix.with_suffix(".log").read_text()
                )
            manifest["run_order"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            result.check_returncode()
            read_run(prefix.with_suffix(".jsonl"), mode, depth)
            if identity() != identities:
                raise ValueError("measurement identity changed during capture")
            export = [
                "docker",
                "run",
                "--rm",
                "-u",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{ROOT}:{ROOT}",
                "-w",
                str(ROOT),
                "--entrypoint",
                "nsys",
                identities["candidate_image"],
                "export",
                "--type",
                "sqlite",
                "--output",
                str(prefix.with_suffix(".sqlite")),
                str(prefix.with_suffix(".nsys-rep")),
            ]
            with prefix.with_suffix(".export.log").open("w") as log:
                subprocess.run(export, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest["status"] = "COMPLETE"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{directory}: COMPLETE", flush=True)


def observed_ratio(candidate_ms: float, llama_ms: float, work: int) -> dict:
    """Describe the observed ratio without a confidence or percentile claim."""
    if (
        any(
            not math.isfinite(value) or value <= 0 for value in (candidate_ms, llama_ms)
        )
        or work <= 0
    ):
        raise ValueError("positive timings and work required")
    ratio = llama_ms / candidate_ms
    return {
        "qw38_ms": candidate_ms,
        "llama_ms": llama_ms,
        "ratio": ratio,
        "observed_parity": "met" if ratio >= 1 else "unmet",
        "runs_per_engine": 1,
        "qw38_tokens_per_second": work * 1000 / candidate_ms,
        "llama_tokens_per_second": work * 1000 / llama_ms,
    }


def summarize() -> None:
    """Report every required row, deriving prefill from the request's TTFT."""
    directory = CACHE / "single-run"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE" or manifest["identities"] != identity():
        raise ValueError("incomplete or stale capture")
    summaries = {}
    for mode, depth in WORKLOADS:
        arms = {
            engine: read_run(directory / f"{mode}-{depth}-{engine}.jsonl", mode, depth)
            for engine in ("qw38", "llama")
        }
        candidate, llama = arms["qw38"][1], arms["llama"][1]
        metrics = {
            "total": observed_ratio(candidate["total_ms"], llama["total_ms"], 128)
        }
        if mode == "request":
            metrics["ttft"] = observed_ratio(
                candidate["ttft_ms"], llama["ttft_ms"], depth
            )
            metrics["decode_tail"] = observed_ratio(
                candidate["tail_ms"], llama["tail_ms"], 127
            )
            summaries[f"prefill-{depth}"] = {
                "metrics": {"total": metrics["ttft"]},
                "source": f"request-{depth} prompt phase",
            }
        summaries[f"{mode}-{depth}"] = {
            "metrics": metrics,
            "different_output_positions": sum(
                a != b
                for a, b in zip(
                    candidate["output_ids"], llama["output_ids"], strict=True
                )
            ),
            "cold_setup_ms": {
                engine: pair[0]["cold_setup_ms"] for engine, pair in arms.items()
            },
            "resident_used_bytes": {
                engine: pair[0]["initial_free_bytes"] - pair[1]["free_bytes"]
                for engine, pair in arms.items()
            },
        }
    result = {
        "status": "TIMING_COMPLETE",
        "rows": summaries,
        "protocol": manifest["protocol"],
        "instrumentation": manifest["instrumentation"],
        "statistics": "one observation per engine/row; no median, p99, confidence interval or repeated-run inference",
        "profile_memory_status": "allocation peaks and kernel attribution reported separately from these timing summaries",
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    """Capture or summarize the authorized single-run performance protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "summarize"))
    args = parser.parse_args()
    (capture if args.mode == "run" else summarize)()


if __name__ == "__main__":
    main()
