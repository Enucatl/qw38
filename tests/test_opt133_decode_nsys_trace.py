"""Host tests for OPT-133 Nsight Systems decode-window tracing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt133_decode_nsys_trace import (
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    NATIVE,
    NSYS_BIN,
    PARENT,
    PHASES,
    PREFIXES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    REQUIRED_SIDECAR_KEYS,
    SELECTOR,
    WINDOW_TOKENS,
    WINDOWS,
    answers_from_phases,
    family_plan,
    nsys_profile_command,
    nsys_stats_command,
    nsys_capture_script,
    modern_nsys,
    overhead_pair,
    reconcile_window,
    summarize_nsys_stats,
    validate_fixture,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE_SRC = ROOT / "cuda/opt133_decode_nsys_trace_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
GPUTRACE = ROOT / "tests/data/opt133_nsys_gputrace.json"
CUDAAPI = ROOT / "tests/data/opt133_nsys_cudaapisum.json"
NVTX = ROOT / "tests/data/opt133_nsys_nvtxsum.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-133")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt133_decode_nsys_trace.py").read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-133"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["parent"] == PARENT
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["prefixes"] == list(PREFIXES)
    assert contract["windows"] == list(WINDOWS)
    assert contract["window_tokens"] == WINDOW_TOKENS
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["trace"] == "cuda,nvtx,osrt"
    assert contract["trace_requested"] == "cuda,nvtx,cudart,osrt"
    assert iteration["diagnostics_make_target"] == "cuda-opt133-diagnostics"
    assert iteration["gpu_lock"] == "build/optimization-runs/qw38-gpu.lock"
    assert iteration["image"] == "qw38-cuda:13.0.2"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt133-diagnostics" in makefile
    assert "qw38-cuda-opt133-decode-nsys-trace-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "--workload identity|nsys-window|nsys-baseline" in native
    assert "cudaProfilerStart" in native
    assert "cudaProfilerStop" in native
    assert "opt133.window" in native
    assert "nvtxRangePushA" in native
    assert "QW38_OPT133_GATE" in native
    assert "independently_restored=true" in native
    assert "greedy_sample" in native
    assert "QW38_OPT133_DECODE_NSYS_TRACE_RESULT=" in native
    assert "QW38_OPT133_RECORDS=" in native
    assert "QW38_OPT133_DECODE_NSYS_TRACE_RESULT=" in runner
    assert "from tools.opt125_decode_accounting import" in tool
    assert "classify_disjoint_intervals" in tool
    assert PHASES == (
        "preflight",
        "capture",
        "parse",
        "reconcile",
        "overhead",
        "report",
    )
    validate_future_keep_policy("OPT-133", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    for key in REQUIRED_SIDECAR_KEYS:
        assert key in contract["required_sidecar_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-133")
    preflight = iteration["workloads"]["preflight"]
    capture = iteration["workloads"]["capture"]
    parse = iteration["workloads"]["parse"]
    reconcile = iteration["workloads"]["reconcile"]
    overhead = iteration["workloads"]["overhead"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(capture, "feedback")) == 144
    assert loop_product(workload_for_mode(parse, "feedback")) == 6
    assert loop_product(workload_for_mode(reconcile, "feedback")) == 6
    assert loop_product(workload_for_mode(overhead, "feedback")) == 12
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-133", "feedback", iteration, "capture")
    assert "phase=capture" in described
    assert "loop_product=1" in family_plan("feedback", "preflight")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "claims_throughput=false" in proof
    assert "12-token" in proof
    assert "no full 256-token nsys profile" in proof


def test_nsys_command_line_builder() -> None:
    command = nsys_profile_command(
        "build/optimization-runs/opt133/trace-128-early",
        [
            "./build/qw38-cuda-opt133-decode-nsys-trace-test",
            "--workload",
            "nsys-window",
            "--window",
            "early",
            "--prefix",
            "128",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
    )
    assert command[0] == NSYS_BIN
    assert command[1] == "profile"
    assert "-o" in command
    assert "--force-overwrite" in command
    assert "true" in command
    assert "--trace=cuda,nvtx,osrt" in command
    if modern_nsys():
        assert "--capture-range=cudaProfilerApi" in command
        assert "--capture-range-end=repeat" in command
    else:
        assert "--capture-range=nvtx" in command
        assert "--nvtx-capture=opt133.window" in command
        assert "--capture-range-end=stop" in command
        assert "--kill=none" in command
    script = nsys_capture_script(
        "build/optimization-runs/opt133/trace-128-early",
        [
            "./build/qw38-cuda-opt133-decode-nsys-trace-test",
            "--workload",
            "nsys-window",
        ],
        session="opt133d128early",
    )
    assert "QdstrmImporter" in script
    assert "qdstrm" in script
    assert "nsys" in script and "launch" in script
    assert "--session-new=opt133d128early" in script
    assert "QW38_OPT133_GATE=" in script
    assert "cuda-graph-trace=graph" in script
    assert "result.txt" in script
    assert command[-1].endswith("Qwen3.8-27B-Q4_K_M.gguf")
    stats = nsys_stats_command(
        "gputrace", "build/optimization-runs/opt133/trace-128-early.nsys-rep"
    )
    assert stats[0] == NSYS_BIN
    assert stats[1:3] == [
        "stats",
        "--report",
    ]
    assert "gputrace" in stats
    assert "--format" in stats
    assert "json" in stats


def test_nsys_stats_exemplar_parser() -> None:
    gputrace = _json(GPUTRACE)
    cudaapi = _json(CUDAAPI)
    nvtx = _json(NVTX)
    summary = summarize_nsys_stats(gputrace, cudaapi, nvtx)
    assert summary["from_parsed_trace"] is True
    assert summary["gpu_busy_ms"] == 8.0
    assert summary["gpu_idle_ms"] == 1.5
    assert summary["capture_window_ms"] == 9.5
    assert summary["cuda_api_ms"] == 3.9
    assert summary["graph_launch_count"] == 8
    assert summary["graph_launch_ms"] == 2.0
    assert summary["memcpy_ms"] == 1.4
    assert summary["sync_ms"] == 1.5
    assert summary["nvtx"]["ffn"] > 0.0
    assert summary["nvtx"]["graph"] > 0.0


def test_reconcile_math_against_synthetic_opt125() -> None:
    records = [
        {
            "role": "ffn_mmq",
            "attribution_role": "member",
            "start_ms": 0.0,
            "end_ms": 8.0,
            "complete_work_ms": 8.0,
        },
        {
            "role": "attention_core",
            "attribution_role": "member",
            "start_ms": 18.0,
            "end_ms": 20.0,
            "complete_work_ms": 2.0,
        },
    ]
    opt125 = {
        "unobserved_ms": 42.1,
        "device_active_ms": 33.2,
        "device_inactive_proven_ms": 0.0,
        "wall_ms": 75.3,
        "tokens": 12,
    }
    parsed = {
        "gpu_idle_ms": 5.0,
        "gpu_busy_ms": 90.0,
        "cuda_api_ms": 8.0,
        "capture_window_ms": 95.0,
        "from_parsed_trace": True,
    }
    row = reconcile_window(opt125, parsed, records)
    assert row["nsys_gpu_idle_ms"] == 5.0
    assert row["delta_idle_minus_unobserved_ms"] == 5.0 - 42.1
    assert row["explained_by_idle_ms"] == 5.0
    assert row["explained_by_api_ms"] == 8.0
    assert row["unresolved_ms"] == 42.1 - 5.0 - 8.0
    assert row["fraction_unresolved"] > 0.0
    assert row["profiler"] == "nsight_systems"
    assert row["stub_relabel_avoided"] is True
    assert row["intervals"]["profiler"] == "nsight_systems"
    assert row["intervals"]["device_inactive_proven_ms"] == 5.0
    stub = reconcile_window(opt125, {"from_parsed_trace": False}, records)
    assert stub["profiler"] == "cuda_event_engine_attribution"
    assert stub["nsys_gpu_idle_ms"] is None
    assert stub["explained_by_idle_ms"] == 0.0
    assert stub["intervals"]["device_inactive_proven_ms"] == 0.0


def test_overhead_and_report_validator() -> None:
    pair = overhead_pair(100.0, 112.0)
    assert pair["nsys_overhead_ms"] == 12.0
    assert pair["nsys_overhead_ratio"] == 1.12
    missing = overhead_pair(None, 10.0)
    assert missing["nsys_overhead_ms"] is None
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert REPORT.parent.as_posix().endswith("opt133-decode-nsys-trace")
    answers = answers_from_phases(
        {
            "windows": {
                "d128-early": {
                    "from_parsed_trace": True,
                    "explained_by_idle_ms": 5.0,
                    "explained_by_api_ms": 8.0,
                    "unresolved_ms": 29.1,
                    "opt125_unobserved_ms": 42.1,
                }
            }
        },
        {"mean_nsys_overhead_ms": 12.0, "windows": {"d128-early": pair}},
    )
    assert answers["proven_gpu_idle"]["mean_explained_by_idle_ms"] == 5.0
    assert answers["nsys_wrapper_overhead"]["mean_nsys_overhead_ms"] == 12.0
    proof = " ".join(_json(CONTRACT)["proof_limit"]).casefold()
    assert "no production selector" in proof
    assert "nsight_systems only with parsed gpu_idle_ms" in proof
