"""Host tests for OPT-132 combined decode stack admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt132_combined_decode import (
    CANDIDATE,
    COMBINATION_GRAPHS,
    CONTRACT,
    CONTROL_GRAPHS,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    NATIVE,
    PAIR_COUNT,
    PARENT,
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
NATIVE_SRC = ROOT / "cuda/opt132_combined_decode_test.cu"
HEADER = ROOT / "cuda/full_scheduler.h"
GRAPH = ROOT / "cuda/execution_graph_path.cuh"
ATTENTION = ROOT / "cuda/attention_decode_path.cuh"
PACKED = ROOT / "cuda/opt120_packed_kv.cuh"
REQUANT = ROOT / "cuda/opt121_weight_requant.cuh"
GDN = ROOT / "cuda/opt122_gdn_state_precision.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-132")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-132"
    assert contract["parent"] == PARENT
    assert contract["historical_control"] == "post113_selected"
    assert contract["candidate"] == CANDIDATE
    assert contract["keepers"] == ["OPT-118", "OPT-119", "OPT-126+127"]
    assert contract["independent_survivor_unit"] == "OPT-126+127"
    assert contract["rejected"]["OPT-128"]["verdict"] == "no_material_opportunity"
    assert contract["rejected"]["OPT-130"]["verdict"] == "reject"
    assert contract["rejected"]["OPT-131"]["verdict"] == "no_material_opportunity"
    assert contract["interaction_matrix"] == [
        "control",
        "combination",
        "combination_minus_opt126_127",
    ]
    assert contract["graph_attention_interaction"] == ("not_applicable_opt130_rejected")
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["require_candidate_nll"] is True
    assert contract["quality_budget_ratchets"] is False
    assert contract["opt113_numbers_historical_only"] is True
    assert contract["opt056_plus_5pct_historical_only"] is True
    assert contract["metrics"] == ["decode_only", "complete_request"]
    assert iteration["diagnostics_make_target"] == "cuda-opt132-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt132-diagnostics" in makefile
    assert "qw38-cuda-opt132-combined-decode-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedLazyOutputMaterialization" in header
    assert "kSelectedMixerNormQ8Fusion" in header
    assert "kSelectedPollWithoutDeviceSync = false" in header
    assert "kSelectedDeferElapsedEventSync = false" in header
    assert "--workload identity|recapture-proof|cancellation|same-math|" in native
    assert "independently_restored=true" in native
    assert "metric_clock_starts_after_prefill_for_decode_only" in native
    assert "QW38_OPT132_COMBINED_DECODE_RESULT=" in native
    assert "QW38_OPT132_NATIVE_COUNTS=" in runner
    assert "QW38_OPT132_COMBINED_DECODE_RESULT=" in runner
    assert "decode_only_tok_s" in native
    assert "request_tok_s" in native
    validate_future_keep_policy("OPT-132", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-132")
    identity = iteration["workloads"]["identity"]
    recapture = iteration["workloads"]["recapture-proof"]
    api = iteration["workloads"]["api"]
    p4096 = iteration["workloads"]["p4096"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    loo = iteration["workloads"]["leave-one-out"]
    long_ctx = iteration["workloads"]["long-context"]
    activity = iteration["workloads"]["activity"]
    assert loop_product(workload_for_mode(identity, "feedback")) == 2
    assert loop_product(workload_for_mode(recapture, "feedback")) == 32
    assert loop_product(workload_for_mode(api, "feedback")) == 2
    assert loop_product(workload_for_mode(p4096, "acceptance")) == 26
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(loo, "acceptance")) == 48
    assert loop_product(workload_for_mode(long_ctx, "acceptance")) == 96
    assert loop_product(workload_for_mode(activity, "acceptance")) == 1024
    described = describe_plan("OPT-132", "feedback", iteration, "identity")
    assert "phase=identity" in described
    assert "loop_product=2" in family_plan("feedback", "identity")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 10 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof
    assert "aggregate complete-request ci lower >1" in proof
    assert "opt-130" in proof
    assert "no new optimization" in proof
    assert "opt-126+127 is one valid unit" in proof


def test_keep_requires_opt116_and_opt125_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-132")
    assert contract["target_workloads"] == ["d128", "d2048"]
    assert "p4096" in contract["non_target_workloads"]
    assert "d8192" in contract["non_target_workloads"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "quality" in iteration["modes"]["acceptance"]["tier_sequence"]
    assert iteration["workloads"]["two-k"]["target"].endswith("prefill-2k-parity-test")
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt132-combined-decode")
    assert FIXTURE.name == "opt132_combined_decode.json"
    assert PARENT == "post124_combined_opt118_opt119"
    assert contract["removal_rules_frozen_before_results"] is True
    assert contract["aggregate_metric"] == "complete_request"


def test_rejected_paths_cannot_enter_combination() -> None:
    header = HEADER.read_text(encoding="utf-8")
    graphs = GRAPH.read_text(encoding="utf-8")
    attention = ATTENTION.read_text(encoding="utf-8")
    packed = PACKED.read_text(encoding="utf-8")
    requant = REQUANT.read_text(encoding="utf-8")
    gdn = GDN.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    assert f'kSelectedExecutionGraphPath[] = "{COMBINATION_GRAPHS}"' in graphs or (
        f'kSelectedExecutionGraphPath[] = "{CONTROL_GRAPHS}"' in graphs
    )
    assert "kSelectedPackedKvFormat = PackedKvFormat::kDenseBf16" in packed
    assert 'kSelectedWeightRequantConfig[] = "none"' in requant
    assert "kFp32" in gdn
    assert "kSelectedVec128NParts = 16" in attention
    assert "kSelectedDecodeAttentionVerifiedMax = 4096" in attention
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in attention
    assert "kSelectedPollWithoutDeviceSync = false" in header
    assert "kSelectedDeferElapsedEventSync = false" in header
    assert "opt130_selector_excluded" in native
    assert "opt128_pins_excluded" in native
    assert "opt131_fusion_excluded" in native
    assert "kLegalExecutionGraphFfnOnly" in native
    assert "kLegalExecutionGraphDecodeSegments8" in native


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-132"
    assert payload["parent"] == PARENT
    assert payload["historical_control"] == "post113_selected"
    assert payload["keepers"] == ["OPT-118", "OPT-119", "OPT-126+127"]
    assert payload.get("rejected_no_leak") is True
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["shipping_execution_graphs"] == COMBINATION_GRAPHS
    else:
        assert payload["verdict"] in {
            "reject",
            "quality_blocked",
            "inconclusive",
            "retained_control",
        }
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"keep", "reject", "retained_control"}:
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("opt058_invoked") is True
    performance = payload.get("performance") or {}
    assert "p4096" in performance
    assert "d128" in performance
    assert "d2048" in performance
    assert "aggregate" in performance
    assert "decode_only" in (performance.get("d128") or {})
    assert "request" in (performance.get("d128") or {})
    assert "headroom" in payload
    assert "activity" in payload
