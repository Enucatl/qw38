"""Host tests for OPT-126 stable decode-graph inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt126_stable_graph_inputs import (
    CANDIDATE,
    CONTRACT,
    FIXTURE,
    INVENTORY,
    ITERATION,
    NATIVE,
    PARENT,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    family_plan,
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
NATIVE_SRC = ROOT / "cuda/opt126_stable_graph_inputs_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
LAUNCH_STATE = ROOT / "cuda/decode_launch_state.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-126")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    launch = LAUNCH_STATE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    apply = scheduler.split("SchedulerGraphs::apply_segment_launch_params")[1]
    apply = apply.split("SchedulerGraphs::capture_decode_segments")[0]
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-126"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["parent_execution_graphs"] == PARENT
    assert contract["candidate"] == CANDIDATE
    assert contract["selected_execution_graph_path"] == PARENT
    assert contract["gdn_parity_design"] == "stable_pingpong_committed_slot"
    assert contract["topology_count"] == 2
    assert contract["decode_segment_graph_count"] == 16
    assert contract["parity_positions"] == [0, 127, 1023, 1024, 2047, 131071]
    assert contract["opaque_byte_patching"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt126-diagnostics"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt126-diagnostics" in makefile
    assert "qw38-cuda-opt126-stable-graph-inputs-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS" in makefile
    assert "-DQW38_DIAGNOSTIC_TRACE" in makefile
    assert "--workload inventory|positions|isolation|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "decode_segments8" in native
    assert "ffn_only" in native
    assert "QW38_OPT126_STABLE_GRAPH_INPUTS_RESULT=" in native
    assert "QW38_OPT126_NATIVE_COUNTS=" in native
    assert "QW38_OPT126_NATIVE_COUNTS=" in runner
    assert "QW38_OPT126_STABLE_GRAPH_INPUTS_RESULT=" in runner
    assert "struct DecodeLaunchState" in launch
    assert "stable_pingpong_committed_slot" in launch
    assert "commit_gdn_slot" in scheduler
    assert "upload_decode_launch_state" in scheduler
    assert "patch_segment_kernel_params" not in apply
    assert "kDecodeGraphTopologyCount" in scheduler
    assert PHASES == (
        "preflight",
        "inventory",
        "positions",
        "isolation",
        "restore",
        "same-math",
        "costs",
        "report",
    )
    validate_future_keep_policy("OPT-126", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-126")
    preflight = iteration["workloads"]["preflight"]
    inventory = iteration["workloads"]["inventory"]
    positions = iteration["workloads"]["positions"]
    isolation = iteration["workloads"]["isolation"]
    restore = iteration["workloads"]["restore"]
    same = iteration["workloads"]["same-math"]
    costs = iteration["workloads"]["costs"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(inventory, "feedback")) == 1
    assert loop_product(workload_for_mode(positions, "feedback")) == 12
    assert loop_product(workload_for_mode(isolation, "feedback")) == 2
    assert loop_product(workload_for_mode(restore, "feedback")) == 1
    assert loop_product(workload_for_mode(same, "feedback")) == 16
    assert loop_product(workload_for_mode(costs, "feedback")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-126", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=1" in family_plan("feedback", "inventory")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "claims_throughput=false" in proof
    assert "ffn_only" in proof
    assert "opaque" in proof


def test_keep_is_not_a_speed_claim() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-126")
    assert contract["claims_throughput"] is False
    assert iteration["claims_throughput"] is False
    assert "d128" not in iteration["workloads"]
    assert "quality" not in iteration["workloads"]
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt126-stable-graph-inputs")
    assert FIXTURE.name == "opt126_stable_graph_inputs.json"
    assert INVENTORY.name == "INVENTORY.md"


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["claims_throughput"] is False
    assert payload["claims_performance_improvement"] is False
    assert payload["shipping_execution_graphs"] == PARENT
    assert payload["production_kept"] is False
    assert REPORT.is_file()
    assert INVENTORY.is_file()
    report = REPORT.read_text(encoding="utf-8")
    assert "claims_throughput=false" in report
    assert "Launch-state" in report or "launch_state" in report
