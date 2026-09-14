"""Host tests for OPT-128 host submission and synchronization stalls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt128_host_stalls import (
    CANDIDATES,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    HEADER,
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
NATIVE_SRC = ROOT / "cuda/opt128_host_stalls_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt128_host_stalls.py"
ENGINE = ROOT / "src/engine.cpp"
SERVER = ROOT / "src/server_generation.cpp"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-128")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    server = SERVER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-128"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == PARENT
    assert contract["candidates"] == list(CANDIDATES)
    assert contract["max_changes"] == 2
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["require_candidate_nll"] is False
    assert "complete_request_ab" in contract["require_candidate_nll_when"]
    assert "no_material_opportunity" in contract["verdicts"]
    assert iteration["diagnostics_make_target"] == "cuda-opt128-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt128-diagnostics" in makefile
    assert "qw38-cuda-opt128-host-stalls-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedPollWithoutDeviceSync" in header
    assert "kSelectedDeferElapsedEventSync" in header
    assert "struct HostStallTimings" in header
    assert "poll_eval_control" in scheduler
    assert "defer_elapsed_event_sync_enabled" in scheduler
    assert "--workload profile|lifetime|cancellation|logits-consumer|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "QW38_OPT128_HOST_STALLS_RESULT=" in native
    assert "QW38_OPT128_NATIVE_COUNTS=" in native
    assert "QW38_OPT128_NATIVE_COUNTS=" in runner
    assert "QW38_OPT128_HOST_STALLS_RESULT=" in runner
    assert "no_material_opportunity" in tool
    assert "session->sample" in server or "session->eval" in server
    assert "greedy_sample" in engine
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    validate_future_keep_policy("OPT-128", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-128")
    profile = iteration["workloads"]["profile"]
    cancel = iteration["workloads"]["cancellation"]
    freeze = iteration["workloads"]["freeze"]
    screen = iteration["workloads"]["screen-d128"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(profile, "feedback")) == 80
    assert loop_product(workload_for_mode(cancel, "feedback")) == 2
    assert loop_product(workload_for_mode(freeze, "feedback")) == 2
    assert loop_product(workload_for_mode(screen, "feedback")) == 16
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    described = describe_plan("OPT-128", "feedback", iteration, "profile")
    assert "phase=profile" in described
    assert "loop_product=80" in family_plan("feedback", "profile")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "shifted wait is not a removed wait" in proof
    assert "7200 is a ceiling" in proof
    assert "opt-127 kept decode_segments8" in proof


def test_keep_requires_opt125_gates_when_admitted() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-128")
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
    assert REPORT.parent.as_posix().endswith("opt128-host-stalls")
    assert FIXTURE.name == "opt128_host_stalls.json"


def test_pins_match_keep_state() -> None:
    header = HEADER.read_text(encoding="utf-8")
    poll = False
    defer = False
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        poll = bool(payload.get("shipping_poll_without_device_sync"))
        defer = bool(payload.get("shipping_defer_elapsed_event_sync"))
    assert f"kSelectedPollWithoutDeviceSync = {'true' if poll else 'false'};" in header
    assert f"kSelectedDeferElapsedEventSync = {'true' if defer else 'false'};" in header
    contract = _json(CONTRACT)
    assert contract["selected_poll_without_device_sync"] == poll
    assert contract["selected_defer_elapsed_event_sync"] == defer


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-128"
    assert payload["parent"] == PARENT_STACK
    assert payload["verdict"] in {
        "keep",
        "reject",
        "no_material_opportunity",
        "quality_blocked",
        "inconclusive",
    }
    quality = payload.get("quality") or {}
    if payload.get("verdict") == "keep":
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
        assert payload.get("production_kept") is True
    if payload.get("verdict") == "no_material_opportunity":
        assert payload["tok_s_delta"]["speedup"] == 0.0
        assert quality.get("required") is False
        assert payload.get("shipping_poll_without_device_sync") is False
        assert payload.get("shipping_defer_elapsed_event_sync") is False
    assert REPORT.is_file()
