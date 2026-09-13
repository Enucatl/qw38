"""Host tests for OPT-121 selective weight requant keep/reject."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt121_weight_traffic import (
    CANDIDATES,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    NATIVE,
    PAIR_COUNT,
    PARENT,
    REPORT,
    REQUIRE_CANDIDATE_NLL,
    REQUIRED_FIXTURE_KEYS,
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
NATIVE_SRC = ROOT / "cuda/opt121_weight_requant_test.cu"
HEADER = ROOT / "cuda/opt121_weight_requant.cuh"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
QUANT = ROOT / "src/quant.cpp"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt121_weight_traffic.py"
QUALITY_SRC = ROOT / "cuda/opt058_quality_baseline_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-121")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    quant = QUANT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    quality = QUALITY_SRC.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-121"
    assert contract["parent"] == PARENT
    assert contract["parent_execution_graphs"] == "ffn_only"
    assert contract["primary_target"] == "decode"
    assert contract["max_changes"] == 1
    assert contract["max_configs"] == 3
    assert contract["candidates"] == list(CANDIDATES)
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["non_target_point_ratio_min"] == 0.98
    assert contract["decode_p95_ratio_max"] == 1.05
    assert contract["require_candidate_nll"] is True
    assert REQUIRE_CANDIDATE_NLL is True
    assert iteration["diagnostics_make_target"] == "cuda-opt121-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt121-diagnostics" in makefile
    assert "qw38-cuda-opt121-weight-traffic-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert 'kSelectedWeightRequantConfig[] = "none"' in header
    assert "WeightRequantConfig::kQ8ToQ4K" in header
    assert "fp4_study" in header
    assert "apply_weight_requant" in scheduler
    assert "g_q4_mmq_aligned_override" in scheduler
    assert "encode_q4_k" in quant
    assert "--workload reference|inventory|greedy-parity|api|" in native
    assert "independently_restored=true" in native
    assert "QW38_OPT121_WEIGHT_TRAFFIC_RESULT=" in native
    assert "QW38_OPT121_NATIVE_COUNTS=" in runner
    assert "QW38_OPT121_WEIGHT_TRAFFIC_RESULT=" in runner
    assert "require_candidate_nll" in tool
    assert "same_math_copied_from_control" in tool
    assert "--weight-requant" in quality
    validate_future_keep_policy("OPT-121", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-121")
    inventory = iteration["workloads"]["inventory"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    assert loop_product(workload_for_mode(inventory, "feedback")) == 3
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    described = describe_plan("OPT-121", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=3" in family_plan("feedback", "inventory")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 10 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof
    assert "primary targets are d128 and d2048" in proof
    assert "memory-only" in proof
    assert "do not relabel q4_k bits as fp4" in proof


def test_keep_requires_opt115_and_opt116_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-121")
    assert contract["target_workloads"] == ["d128", "d2048"]
    assert contract["non_target_workloads"] == ["p4096"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert iteration["workloads"]["long-cache"]["target"].endswith(
        "opt116-generated-quality-test"
    )
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt121-weight-traffic")
    assert FIXTURE.name == "opt121_weight_traffic.json"
    assert PARENT == "post113_selected"


def test_pins_match_keep_state() -> None:
    kept = "none"
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        if payload.get("production_kept"):
            kept = str(payload.get("selected_weight_requant") or "none")
    header = HEADER.read_text(encoding="utf-8")
    assert f'kSelectedWeightRequantConfig[] = "{kept}"' in header
    contract = _json(CONTRACT)
    assert contract["selected_weight_requant"] == kept


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-121"
    assert payload["parent"] == "post113_selected"
    assert payload["primary_target"] == "decode"
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["selected_weight_requant"] != "none"
    else:
        assert payload["verdict"] in {
            "reject",
            "zero_redundancy",
            "quality_blocked",
            "inconclusive",
        }
        assert payload["selected_weight_requant"] == "none"
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"keep", "reject", "quality_blocked"}:
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("opt058_invoked") is True
