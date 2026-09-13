"""Host tests for OPT-127 recapture-free decode-segment replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt127_recapture_free_replay import (
    CANDIDATE,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    NATIVE,
    PAIR_COUNT,
    PARENT,
    PARENT_STACK,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    WARMUPS,
    family_plan,
    pair_order,
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
NATIVE_SRC = ROOT / "cuda/opt127_recapture_free_replay_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
HEADER = ROOT / "cuda/full_scheduler.h"
LAUNCH_STATE = ROOT / "cuda/decode_launch_state.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-127")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    launch = LAUNCH_STATE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    apply = scheduler.split("SchedulerGraphs::apply_segment_launch_params")[1]
    apply = apply.split("SchedulerGraphs::capture_decode_segments")[0]
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-127"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == PARENT
    assert contract["candidate"] == CANDIDATE
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["non_target_point_ratio_min"] == 0.98
    assert contract["decode_p95_ratio_max"] == 1.05
    assert contract["require_candidate_nll"] is True
    assert contract["metrics"] == ["decode_only", "complete_request"]
    assert contract["coarser_candidate"] == "not_admitted"
    assert iteration["diagnostics_make_target"] == "cuda-opt127-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt127-diagnostics" in makefile
    assert "qw38-cuda-opt127-recapture-free-replay-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "struct GraphLifecycleCounts" in header
    assert "graph_capture_count" in header
    assert "graph_instantiate_count" in header
    assert "graph_upload_count" in header
    assert "kDecodeGraphInvalidationPolicy" in launch
    assert "++graph_capture_count_" in scheduler
    assert "++graph_instantiate_count_" in scheduler
    assert "++graph_upload_count_" in scheduler
    assert "patch_segment_kernel_params" not in apply
    assert "--workload recapture-proof|cancellation|same-math|handoff|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "decode_only_tok_s" in native
    assert "request_tok_s" in native
    assert "QW38_OPT127_RECAPTURE_FREE_REPLAY_RESULT=" in native
    assert "QW38_OPT127_NATIVE_COUNTS=" in native
    assert "QW38_OPT127_NATIVE_COUNTS=" in runner
    assert "QW38_OPT127_RECAPTURE_FREE_REPLAY_RESULT=" in runner
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    validate_future_keep_policy("OPT-127", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-127")
    recapture = iteration["workloads"]["recapture-proof"]
    cancel = iteration["workloads"]["cancellation"]
    same = iteration["workloads"]["same-math"]
    handoff = iteration["workloads"]["handoff"]
    screen = iteration["workloads"]["screen-d128"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(recapture, "feedback")) == 32
    assert loop_product(workload_for_mode(cancel, "feedback")) == 2
    assert loop_product(workload_for_mode(same, "feedback")) == 16
    assert loop_product(workload_for_mode(handoff, "feedback")) == 2
    assert loop_product(workload_for_mode(screen, "feedback")) == 16
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    described = describe_plan("OPT-127", "feedback", iteration, "recapture-proof")
    assert "phase=recapture-proof" in described
    assert "loop_product=32" in family_plan("feedback", "recapture-proof")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 5 paired ab/ba screen" in proof
    assert "7200 is a ceiling" in proof
    assert "decode-only and complete-request" in proof


def test_keep_requires_opt125_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-127")
    assert contract["target_workloads"] == ["d128", "d2048"]
    assert contract["non_target_workloads"] == ["p4096"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert (
        "build/qw38-cuda-opt058-quality-baseline-test"
        in iteration["modes"]["acceptance"]["make_targets"]
    )
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt127-recapture-free-replay")
    assert FIXTURE.name == "opt127_recapture_free_replay.json"


def test_pins_match_keep_state() -> None:
    kept = False
    if FIXTURE.is_file():
        kept = bool(_json(FIXTURE).get("production_kept"))
    header = (ROOT / "cuda/execution_graph_path.cuh").read_text(encoding="utf-8")
    expected = CANDIDATE if kept else PARENT
    assert f'kSelectedExecutionGraphPath[] = "{expected}"' in header
    contract = _json(CONTRACT)
    assert contract["selected_execution_graph_path"] == expected


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-127"
    assert payload["parent"] == PARENT_STACK
    assert payload["candidate"] == CANDIDATE
    quality = payload.get("quality") or {}
    if payload.get("mode") == "acceptance":
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["shipping_execution_graphs"] == CANDIDATE
    else:
        assert payload["shipping_execution_graphs"] == PARENT
    assert REPORT.is_file()
