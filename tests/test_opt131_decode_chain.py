"""Host tests for OPT-131 residual decode launch-chain fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt131_decode_chain import (
    CANDIDATES,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    GRAPH_PIN,
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
NATIVE_SRC = ROOT / "cuda/opt131_decode_chain_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
PRIMITIVES = ROOT / "cuda/scheduler_primitives.cu"
GDN = ROOT / "cuda/gdn_step.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt131_decode_chain.py"
ATTN = ROOT / "cuda/attention_decode_path.cuh"
HEADER = ROOT / "cuda/full_scheduler.h"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-131")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    primitives = PRIMITIVES.read_text(encoding="utf-8")
    gdn = GDN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    graph = GRAPH_PIN.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-131"
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
    assert contract["recreate_opt119_norm_q8"] is False
    assert "no_material_opportunity" in contract["verdicts"]
    assert iteration["diagnostics_make_target"] == "cuda-opt131-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt131-diagnostics" in makefile
    assert "qw38-cuda-opt131-decode-chain-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert 'kSelectedExecutionGraphPath[] = "decode_segments8"' in graph
    assert "kSelectedMixerNormQ8Fusion = true" in header
    assert "kSelectedFfnNormQ8Fusion = true" in header
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in attn
    assert "launch_prepare_gdn_gates" in primitives
    assert "launch_gdn_prepare_tiled" in scheduler
    assert "launch_gdn_prepare_tiled" in gdn
    assert "--workload identity|profile|occupancy|lifetime|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "QW38_OPT131_DECODE_CHAIN_RESULT=" in native
    assert "QW38_OPT131_NATIVE_COUNTS=" in native
    assert "QW38_OPT131_NATIVE_COUNTS=" in runner
    assert "QW38_OPT131_DECODE_CHAIN_RESULT=" in runner
    assert "no_material_opportunity" in tool
    assert "opt109" in tool
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    validate_future_keep_policy("OPT-131", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-131")
    profile = iteration["workloads"]["profile"]
    cancel = iteration["workloads"]["cancellation"]
    freeze = iteration["workloads"]["freeze"]
    screen = iteration["workloads"]["screen-d128"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    occupancy = iteration["workloads"]["occupancy"]
    equivalence = iteration["workloads"]["equivalence"]
    assert loop_product(workload_for_mode(profile, "feedback")) == 16
    assert loop_product(workload_for_mode(cancel, "feedback")) == 2
    assert loop_product(workload_for_mode(freeze, "feedback")) == 2
    assert loop_product(workload_for_mode(occupancy, "feedback")) == 20
    assert loop_product(workload_for_mode(equivalence, "feedback")) == 4
    assert loop_product(workload_for_mode(screen, "feedback")) == 16
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    described = describe_plan("OPT-131", "feedback", iteration, "profile")
    assert "phase=profile" in described
    assert "loop_product=16" in family_plan("feedback", "profile")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "reuse opt-119" in proof
    assert "opt-109" in proof
    assert "7200 is a ceiling" in proof
    assert "opt-127 kept decode_segments8" in proof
    assert "kernel-only speedup is not a keep" in proof


def test_keep_requires_opt125_gates_when_admitted() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-131")
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
    assert REPORT.parent.as_posix().endswith("opt131-decode-chain")
    assert FIXTURE.name == "opt131_decode_chain.json"


def test_parent_pins_unchanged() -> None:
    graph = GRAPH_PIN.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    assert 'kSelectedExecutionGraphPath[] = "decode_segments8"' in graph
    assert "kSelectedPollWithoutDeviceSync = false;" in header
    assert "kSelectedDeferElapsedEventSync = false;" in header
    assert "kSelectedMixerNormQ8Fusion = true" in header
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in attn
    assert "kSelectedVec128NParts = 16" in attn


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-131"
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
        assert payload.get("shipping_decode_chain_fusion") == "none"
    assert REPORT.is_file()
