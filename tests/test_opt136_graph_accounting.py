"""Host tests for the OPT-136 graph-accounting harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt136_graph_accounting import (
    ARMS,
    CAPACITY,
    CONTRACT,
    DECODE_TOKENS,
    ENGINES,
    FIXTURE,
    ITERATION,
    NATIVE,
    PHASES,
    PREFIXES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    WINDOW_STEPS,
    WINDOW_TOKENS,
    WINDOWS,
    family_plan,
    nsys_profile_command,
    validate_fixture,
)
from tools.performance_evidence import historical_reconciliation_invalid
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE_SRC = ROOT / "cuda/opt136_matched_decode_profile_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
LLAMA_SRC = ROOT / "tools/llama_authority/opt136_decode_profile.cpp"
CURSOR_CHECKLIST = (
    ROOT
    / ".cursor/skills/run-ledger-task-cursor/references/performance-evidence-checklist.md"
)
CODEX_CHECKLIST = (
    ROOT
    / ".agents/skills/run-ledger-task-codex/references/performance-evidence-checklist.md"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-136")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    llama = LLAMA_SRC.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt136_graph_accounting.py").read_text(encoding="utf-8")
    evidence_lib = (ROOT / "tools/performance_evidence.py").read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert LLAMA_SRC.is_file()
    assert contract["task"] == "OPT-136"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["prefixes"] == list(PREFIXES)
    assert contract["windows"] == list(WINDOWS)
    assert contract["window_tokens"] == WINDOW_TOKENS
    assert contract["decode_eval_tokens"] == DECODE_TOKENS
    assert contract["allocated_capacity"] == CAPACITY
    assert iteration["diagnostics_make_target"] == "cuda-opt136-diagnostics"
    assert iteration["gpu_lock"] == "build/optimization-runs/qw38-gpu.lock"
    assert iteration["image"] == "qw38-cuda:13.0.2"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt136-diagnostics" in makefile
    assert "qw38-cuda-opt136-matched-decode-profile-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "--workload identity|unprofiled|graph-capture|node-capture" in native
    assert "cudaProfilerStart" in native
    assert "opt136.window" in native
    assert "nvtxRangePushA" in native
    assert "kOpt136DiagnosticIdentityMarkers" in native
    assert "QW38_OPT136_GRAPH_ACCOUNTING_RESULT=" in native
    assert "QW38_OPT136_GRAPH_ACCOUNTING_RESULT=" in runner
    assert "context_params.n_ctx" in llama
    assert "131072" in llama
    assert "performance_evidence" in tool
    assert "--validate" in evidence_lib
    assert PHASES == (
        "historical",
        "preflight",
        "baseline",
        "capture",
        "analyze",
        "report",
    )
    assert ENGINES == ("quartz", "llama")
    assert ARMS == ("unprofiled", "graph", "node")
    assert WINDOW_STEPS == ((0, 12), (122, 134), (244, 256))
    validate_future_keep_policy("OPT-136", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-136")
    historical = iteration["workloads"]["historical"]
    preflight = iteration["workloads"]["preflight"]
    baseline = iteration["workloads"]["baseline"]
    capture = iteration["workloads"]["capture"]
    analyze = iteration["workloads"]["analyze"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(historical, "feedback")) == 6
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(baseline, "feedback")) == 104
    assert loop_product(workload_for_mode(capture, "feedback")) == 72
    assert loop_product(workload_for_mode(analyze, "feedback")) == 72
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-136", "feedback", iteration, "capture")
    assert "phase=capture" in described
    assert "loop_product=1" in family_plan("feedback", "preflight")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "claims_throughput=false" in proof


def test_nsys_command_includes_graph_and_node_trace() -> None:
    graph = nsys_profile_command(
        "build/optimization-runs/opt136/quartz-d128-graph-r0",
        [
            "./build/qw38-cuda-opt136-matched-decode-profile-test",
            "--workload",
            "graph-capture",
            "--prefix",
            "128",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
        ],
        graph_trace="graph",
    )
    assert "profile" in graph
    assert "--cuda-graph-trace=graph" in graph
    assert "--trace=cuda,nvtx,osrt" in graph
    assert "--sample=none" in graph
    node = nsys_profile_command(
        "build/optimization-runs/opt136/quartz-d128-node-r0",
        ["./build/qw38-cuda-opt136-matched-decode-profile-test"],
        graph_trace="node",
    )
    assert "--cuda-graph-trace=node" in node


def test_idle_reconciliation_is_non_causal() -> None:
    row = historical_reconciliation_invalid(
        old_unobserved_ms=42.1,
        new_idle_ms=188.5,
        reason="test",
        link="evidence/optimization/opt136-graph-accounting/historical-reconciliation.json",
    )
    assert row["causal"] is False
    assert row["current_derived_idle_ms"] is None
    assert row["historical_min_ms"] == 42.1


def test_fixture_and_validate_cli_link() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert REPORT.parent.as_posix().endswith("opt136-graph-accounting")
    cursor = CURSOR_CHECKLIST.read_text(encoding="utf-8")
    codex = CODEX_CHECKLIST.read_text(encoding="utf-8")
    assert "tools/performance_evidence.py --validate" in cursor
    assert "tools/performance_evidence.py --validate" in codex
    cmake = (ROOT / "tools/llama_authority/CMakeLists.txt").read_text(encoding="utf-8")
    assert "qw38-llama-opt136-decode-profile" in cmake
    header = (ROOT / "cuda/engine_attribution.h").read_text(encoding="utf-8")
    assert "kOpt136DiagnosticIdentityMarkers" in header
