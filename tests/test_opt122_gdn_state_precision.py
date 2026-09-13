"""Host tests for OPT-122 GDN state precision bound."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt122_gdn_state_precision import (
    AA_GEO,
    CANDIDATES,
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    HEADER,
    ITERATION,
    NATIVE,
    OPT109_GDN_MS,
    PAIR_COUNT,
    PARENT,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    WARMUPS,
    aa_resolution_rel,
    candidate_savings,
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
NATIVE_SRC = ROOT / "cuda/opt122_gdn_state_precision_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt122_gdn_state_precision.py"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
CHECKPOINT = ROOT / "cuda/checkpoint.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-122")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-122"
    assert contract["parent"] == PARENT
    assert contract["parent_execution_graphs"] == "ffn_only"
    assert contract["max_changes"] == 2
    assert contract["candidates"] == list(CANDIDATES)
    assert contract["convolution_stays_fp32"] is True
    assert contract["update_accumulation_fp32"] is True
    assert contract["no_persistent_fp32_shadow"] is True
    assert contract["selected_gdn_state_format"] == "fp32"
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["require_candidate_nll"] is False
    assert "complete_request_ab" in contract["require_candidate_nll_when"]
    assert "no_material_opportunity" in contract["verdicts"]
    assert iteration["diagnostics_make_target"] == "cuda-opt122-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt122-diagnostics" in makefile
    assert "qw38-cuda-opt122-gdn-state-precision-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedGdnStateFormat = GdnStateFormat::kFp32" in header
    assert "kGdnStateQ8Group = 32" in header
    assert "opt122_recurrence_bf16" in native
    assert "QW38_OPT122_GDN_STATE_PRECISION_RESULT=" in native
    assert "QW38_OPT122_NATIVE_COUNTS=" in runner
    assert "QW38_OPT122_GDN_STATE_PRECISION_RESULT=" in runner
    assert "no_material_opportunity" in tool
    assert "std::swap(session->gdn_recurrent_" in scheduler
    assert "kGdnRecurrentStateValues" in checkpoint
    validate_future_keep_policy("OPT-122", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-122")
    preflight = iteration["workloads"]["preflight"]
    reference = iteration["workloads"]["reference"]
    inventory = iteration["workloads"]["inventory"]
    bound = iteration["workloads"]["bound"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(reference, "feedback")) == 2
    assert loop_product(workload_for_mode(inventory, "feedback")) == 4
    assert loop_product(workload_for_mode(bound, "feedback")) == 2
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-122", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=4" in family_plan("feedback", "inventory")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "a/a resolution" in proof
    assert "7200 is a ceiling" in proof
    assert "recurrence-only speedup never admits" in proof


def test_bound_math_and_aa_resolution() -> None:
    contract = _json(CONTRACT)
    assert contract["aa_geo"] == AA_GEO
    assert abs(aa_resolution_rel() - 0.0001) < 1e-12
    assert contract["opt109_gdn_enclosing_ms"] == OPT109_GDN_MS
    inventory = {
        "dram_fp32_bytes": 317718528,
        "dram_bf16_bytes": 166723584,
        "dram_q8_bytes": 91226112,
        "peak_fp32_ms": 0.177,
        "peak_bf16_ms": 0.093,
        "peak_q8_ms": 0.051,
        "memcpy_rec_fp32_ms": 0.12,
        "memcpy_rec_bf16_ms": 0.07,
        "fp32_recurrence_ms": 1.20,
        "bf16_recurrence_ms": 1.25,
        "gdn_core_ms": 2.22,
        "recurrent_fp32_bytes": 150994944,
        "recurrent_q8_bytes": 40144896,
    }
    winning = dict(inventory)
    winning["fp32_recurrence_ms"] = 0.89
    winning["bf16_recurrence_ms"] = 0.59
    winning["gdn_core_ms"] = 2.01
    winning["memcpy_rec_fp32_ms"] = 0.205
    bf16_win = candidate_savings(winning, "bf16")
    assert bf16_win["leaf_save_ms"] > 0.2
    assert bf16_win["roofline_save_ms"] == 0.0
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt122-gdn-state-precision")
    assert FIXTURE.name == "opt122_gdn_state_precision.json"
    assert "quality" not in PHASES or "report" in PHASES


def test_keep_requires_opt115_gates_when_candidate_admitted() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-122")
    assert contract["target_workloads"] == ["d128", "d2048", "p4096"]
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert contract["decode_p95_ratio_max"] == 1.05
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "complete-request gates apply only if" in proof
    assert contract["gdn_carry_production_traffic"] is False


def test_validate_scaffold_fixture_when_present() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "kSelectedGdnStateFormat = GdnStateFormat::kFp32" in header
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-122"
    assert payload["parent"] == PARENT
    assert payload["selected_gdn_state_format"] == "fp32"
    assert payload["production_kept"] is True
    assert payload["verdict"] in {
        "keep",
        "reject",
        "no_material_opportunity",
        "quality_blocked",
        "inconclusive",
    }
    if payload["verdict"] == "no_material_opportunity":
        assert payload["tok_s_delta"]["speedup"] == 0.0
        assert payload["quality"]["required"] is False
