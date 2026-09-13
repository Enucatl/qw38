"""Host tests for OPT-123 combined pipeline stack sitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt123_combined_stack import (
    CANDIDATE,
    CONTRACT,
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
NATIVE_SRC = ROOT / "cuda/opt123_combined_stack_test.cu"
HEADER = ROOT / "cuda/full_scheduler.h"
GRAPH = ROOT / "cuda/execution_graph_path.cuh"
PACKED = ROOT / "cuda/opt120_packed_kv.cuh"
REQUANT = ROOT / "cuda/opt121_weight_requant.cuh"
GDN = ROOT / "cuda/opt122_gdn_state_precision.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-123")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-123"
    assert contract["parent"] == PARENT
    assert contract["historical_control"] == "post113_selected"
    assert contract["candidate"] == CANDIDATE
    assert contract["keepers"] == ["OPT-118", "OPT-119"]
    assert contract["rejected"]["OPT-117"]["verdict"] == "retain_ffn_only"
    assert contract["rejected"]["OPT-117"]["retain"] == "ffn_only"
    assert contract["rejected"]["OPT-120"]["verdict"] == "quality_blocked"
    assert contract["rejected"]["OPT-121"]["verdict"] == "quality_blocked"
    assert contract["rejected"]["OPT-122"]["verdict"] == "no_material_opportunity"
    assert contract["interaction_matrix"] == [
        "control",
        "combined",
        "opt118_only",
        "opt119_only",
    ]
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["require_candidate_nll"] is True
    assert contract["quality_budget_ratchets"] is False
    assert contract["opt113_numbers_historical_only"] is True
    assert iteration["diagnostics_make_target"] == "cuda-opt123-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt123-diagnostics" in makefile
    assert "qw38-cuda-opt123-combined-stack-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedLazyOutputMaterialization" in header
    assert "kSelectedMixerNormQ8Fusion" in header
    assert 'kSelectedExecutionGraphPath[] = "ffn_only"' in GRAPH.read_text(
        encoding="utf-8"
    )
    assert "kDenseBf16" in PACKED.read_text(encoding="utf-8")
    assert 'kSelectedWeightRequantConfig[] = "none"' in REQUANT.read_text(
        encoding="utf-8"
    )
    assert "kFp32" in GDN.read_text(encoding="utf-8")
    assert "--workload inventory|greedy-parity|api|" in native
    assert "independently_restored=true" in native
    assert "QW38_OPT123_COMBINED_STACK_RESULT=" in native
    assert "QW38_OPT123_NATIVE_COUNTS=" in runner
    assert "QW38_OPT123_COMBINED_STACK_RESULT=" in runner
    assert "rejected OPT-117 leak" in native
    validate_future_keep_policy("OPT-123", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-123")
    inventory = iteration["workloads"]["inventory"]
    greedy = iteration["workloads"]["greedy-parity"]
    api = iteration["workloads"]["api"]
    p4096 = iteration["workloads"]["p4096"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    loo = iteration["workloads"]["leave-one-out"]
    long_ctx = iteration["workloads"]["long-context"]
    assert loop_product(workload_for_mode(inventory, "feedback")) == 4
    assert loop_product(workload_for_mode(greedy, "feedback")) == 2
    assert loop_product(workload_for_mode(api, "feedback")) == 2
    assert loop_product(workload_for_mode(p4096, "acceptance")) == 26
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(loo, "acceptance")) == 48
    assert loop_product(workload_for_mode(long_ctx, "acceptance")) == 96
    described = describe_plan("OPT-123", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=4" in family_plan("feedback", "inventory")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 10 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof
    assert "aggregate throughput lower bound >1" in proof
    assert "rejected paths cannot leak" in proof
    assert "no new optimization" in proof


def test_keep_requires_opt115_and_opt116_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-123")
    assert contract["target_workloads"] == ["p4096", "d128", "d2048"]
    assert "d8192" in contract["non_target_workloads"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert iteration["workloads"]["two-k"]["target"].endswith("prefill-2k-parity-test")
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt123-combined-stack")
    assert FIXTURE.name == "opt123_combined_stack.json"
    assert PARENT == "post113_selected"
    assert contract["removal_rules_frozen_before_results"] is True


def test_rejected_paths_cannot_enter_combination() -> None:
    header = HEADER.read_text(encoding="utf-8")
    graphs = GRAPH.read_text(encoding="utf-8")
    packed = PACKED.read_text(encoding="utf-8")
    requant = REQUANT.read_text(encoding="utf-8")
    gdn = GDN.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    assert 'kSelectedExecutionGraphPath[] = "ffn_only"' in graphs
    assert "kSelectedPackedKvFormat = PackedKvFormat::kDenseBf16" in packed
    assert 'kSelectedWeightRequantConfig[] = "none"' in requant
    assert "kSelectedGdnStateFormat = GdnStateFormat::kFp32" in gdn
    assert "kLegalExecutionGraphDecodeSegments8" not in native or (
        "retains ffn_only" in native
    )
    assert "decode_segments8" not in native
    assert "set_packed_kv" not in native
    assert "q8_to_q4k" not in native
    assert "kQ8Block32" not in native
    assert "apply_stack" in native
    assert "kOpt118" in native
    assert "kCombined" in native
    assert "kSelectedLazyOutputMaterialization" in header
    assert "kSelectedMixerNormQ8Fusion" in header


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-123"
    assert payload["parent"] == "post113_selected"
    assert payload["historical_control"] == "post113_selected"
    assert payload["keepers"] == ["OPT-118", "OPT-119"]
    assert payload.get("rejected_no_leak") is True
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
    else:
        assert payload["verdict"] in {
            "reject",
            "quality_blocked",
            "inconclusive",
        }
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"keep", "reject"}:
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("opt058_invoked") is True
    performance = payload.get("performance") or {}
    assert "p4096" in performance
    assert "d128" in performance
    assert "d2048" in performance
    assert "aggregate" in performance
