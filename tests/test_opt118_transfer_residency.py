"""Host tests for OPT-118 transfer residency keep/reject."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt118_transfer_residency import (
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
NATIVE_SRC = ROOT / "cuda/opt118_transfer_residency_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
HEADER = ROOT / "cuda/full_scheduler.h"
ENGINE = ROOT / "src/engine.cpp"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-118")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-118"
    assert contract["parent"] == "post113_selected"
    assert contract["parent_execution_graphs"] == "ffn_only"
    assert contract["candidate"] == CANDIDATE
    assert contract["max_changes"] == 2
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["non_target_point_ratio_min"] == 0.98
    assert contract["decode_p95_ratio_max"] == 1.05
    assert contract["require_candidate_nll"] is True
    assert contract["gdn_carry_production_traffic"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt118-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt118-diagnostics" in makefile
    assert "qw38-cuda-opt118-transfer-residency-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedLazyOutputMaterialization" in header
    assert "kSelectedOutputCommitOverlap" in header
    assert "greedy_argmax_fp32" in scheduler
    assert "commit_outputs" in scheduler
    assert "materialize_last_outputs" in scheduler
    assert "copy_last_outputs" in engine
    assert "--workload inventory|greedy-parity|api|" in native
    assert "independently_restored=true" in native
    assert "QW38_OPT118_TRANSFER_RESIDENCY_RESULT=" in native
    assert "QW38_OPT118_NATIVE_COUNTS=" in runner
    assert "QW38_OPT118_TRANSFER_RESIDENCY_RESULT=" in runner
    validate_future_keep_policy("OPT-118", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-118")
    inventory = iteration["workloads"]["inventory"]
    greedy = iteration["workloads"]["greedy-parity"]
    api = iteration["workloads"]["api"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    logits = iteration["workloads"]["logits-every-token"]
    assert loop_product(workload_for_mode(inventory, "feedback")) == 2
    assert loop_product(workload_for_mode(greedy, "feedback")) == 2
    assert loop_product(workload_for_mode(api, "feedback")) == 2
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(logits, "acceptance")) == 16
    described = describe_plan("OPT-118", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=2" in family_plan("feedback", "inventory")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 10 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof
    assert "not a throughput keep" in proof


def test_keep_requires_opt115_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-118")
    assert contract["target_workloads"] == ["d128", "d2048"]
    assert contract["non_target_workloads"] == ["p4096"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt118-transfer-residency")
    assert FIXTURE.name == "opt118_transfer_residency.json"
    assert PARENT == "eager_blocking"


def test_inventory_bytes_match_vocab() -> None:
    contract = _json(CONTRACT)
    assert contract["logits_bytes"] == 248320 * 4
    assert contract["hidden_bytes"] == 5120 * 4
    kept = False
    if FIXTURE.is_file():
        kept = bool(_json(FIXTURE).get("production_kept"))
    pin = "true" if kept else "false"
    header = HEADER.read_text(encoding="utf-8")
    assert f"kSelectedLazyOutputMaterialization = {pin}" in header
    assert f"kSelectedOutputCommitOverlap = {pin}" in header
    assert contract["selected_lazy_output_materialization"] is kept
    assert contract["selected_output_commit_overlap"] is kept


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-118"
    assert payload["parent"] == "post113_selected"
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
    else:
        assert payload["verdict"] in {
            "reject",
            "zero_redundancy",
            "quality_blocked",
            "inconclusive",
        }
