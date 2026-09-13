"""Host tests for OPT-111 remaining llama F16 MMA prompt-attention deltas."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from tools.opt075_q4_production_admission import load_json
from tools.opt111_llama_prompt_attention import (
    BASE_ID,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    PRIMITIVE_ROWS,
    PROVENANCE,
    REPORT,
    XOR_ID,
    _run_quality_native,
    decide_verdict,
    evaluate_measured_quality,
    family_plan,
    ppl_ratio,
    primitive_verdict,
    run_quality_phase,
)
from tools.quality.scoring import recurrence_incremental_nll
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
PIPELINE = ROOT / "cuda/fattn_mma_f16.cuh"
PIPELINE_IMPL = ROOT / "cuda/fattn_mma_f16_pipeline.cuh"
ADAPTER = ROOT / "cuda/opt111_llama_prompt_attention.cuh"
NATIVE = ROOT / "cuda/opt111_llama_prompt_attention_test.cu"
RUNNER = ROOT / "tools/opt111_llama_prompt_attention.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-111")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    pin = PIPELINE.read_text(encoding="utf-8")
    impl = PIPELINE_IMPL.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-111"
    assert contract["candidate"] == BASE_ID
    assert contract["control"] == CONTROL_ID
    assert contract["xor_candidate"] == XOR_ID
    assert contract["production_pin"] == CONTROL_ID
    assert contract["stop_if_primitive_loses"] is True
    assert contract["xor_only_if_base_wins_and_hypothesis"] is True
    assert contract["no_tile_swizzle_sweep"] is True
    assert contract["opt079_convert_once_not_new_gain"] is True
    assert contract["require_candidate_nll"] is True
    assert contract["min_saving_ms_per_p4096_attention"] == MIN_SAVING_MS
    assert contract["primitive_rows"] == list(PRIMITIVE_ROWS)
    assert iteration["task"] == "OPT-111"
    assert iteration["diagnostics_make_target"] == "cuda-opt111-diagnostics"
    assert "cuda-opt111-diagnostics" in makefile
    assert "qw38-cuda-opt111-llama-prompt-attention-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    diagnostics_line = makefile.split("cuda-opt111-diagnostics:", 1)[1].split("\n", 1)[
        0
    ]
    assert "qw38-cuda-opt058-quality-baseline-test" in diagnostics_line
    assert "kSelectedAttentionPipelinePath[] =" in pin
    fixture = load_json(FIXTURE)
    shipping = str(fixture.get("shipping_attention_pipeline") or CONTROL_ID)
    assert f'kSelectedAttentionPipelinePath[] = "{shipping}"' in pin
    if fixture.get("production_kept"):
        assert shipping in {BASE_ID, XOR_ID}
    else:
        assert shipping == CONTROL_ID
    assert "opt111_base" in impl and "opt111_xor" in impl
    assert "LlamaLoad" in impl and "XorSwizzle" in impl
    assert 'kControlId[] = "kv_once"' in adapter
    assert "opt111_base" in native and "opt111_xor" in native
    assert "primitive_row" in native
    assert "bank_conflict_hypothesis" in native
    provenance = _json(PROVENANCE)
    assert provenance["llama_revision"] == contract["llama_revision"]
    assert provenance["license"] == "MIT"
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["control"] == CONTROL_ID
    assert fixture["candidate"] == BASE_ID


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-111")
    parity = iteration["workloads"]["parity"]
    primitive = iteration["workloads"]["primitive"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(primitive, "feedback")) == 48
    assert loop_product(workload_for_mode(primitive, "acceptance")) == 156
    described = describe_plan("OPT-111", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "stop if the base path loses" in proof
    assert "20 ms" in proof
    assert "candidate nll" in proof
    assert "xor" in proof
    quality = iteration["workloads"]["quality"]
    assert quality["kind"] == "candidate_nll_if_performance_survivor"
    assert "quality" in iteration["modes"]["release"]["workloads"]
    assert iteration["modes"]["release"]["aggregate_deadline_s"] == 7200


def test_primitive_stop_rule() -> None:
    lose = primitive_verdict(
        {
            32: {"faster": True, "positive": True},
            128: {"faster": True, "positive": True},
            512: {"faster": True, "positive": True},
            2048: {"faster": True, "positive": True},
            4096: {"faster": False, "positive": False},
        },
        mode="feedback",
    )
    assert lose["primitive_win"] is False
    tiled_ignored = primitive_verdict(
        {
            1: {"faster": False, "positive": False},
            32: {
                "faster": True,
                "positive": True,
                "control_mean_ms": 1.0,
                "candidate_mean_ms": 0.8,
            },
            128: {"faster": True, "positive": True},
            512: {"faster": True, "positive": True},
            2048: {"faster": True, "positive": True},
            4096: {
                "faster": True,
                "positive": True,
                "control_mean_ms": 10.0,
                "candidate_mean_ms": 8.0,
            },
        },
        mode="feedback",
    )
    assert tiled_ignored["primitive_win"] is True
    acceptance_needs_ci = primitive_verdict(
        {
            32: {"faster": True, "positive": True},
            128: {"faster": True, "positive": True},
            512: {"faster": True, "positive": True},
            2048: {"faster": True, "positive": True},
            4096: {"faster": True, "positive": False},
        },
        mode="acceptance",
    )
    assert acceptance_needs_ci["primitive_win"] is False
    plan = family_plan("primitive", "feedback")
    assert plan["cases"] == 6
    assert plan["candidates"] == 2
    assert plan["product"] == 48


def test_remaining_differences_and_xor_gate() -> None:
    contract = _json(CONTRACT)
    diffs = contract["remaining_differences"]
    for key in (
        "query_tiling",
        "kv_tiling",
        "shared_memory",
        "mma",
        "softmax",
        "causal_mask",
        "gqa",
        "partition",
        "fixup",
        "combine",
        "kv_conversion_lifetime",
    ):
        assert key in diffs
        assert diffs[key]
    native = NATIVE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "remaining_diff" in native
    assert "matched_primitive_lost" in runner
    assert "bank_conflict_hypothesis" in runner
    impl = PIPELINE_IMPL.read_text(encoding="utf-8")
    assert "fattn_mma_pipeline_opt111_base" in impl
    assert "fattn_mma_pipeline_opt111_xor" in impl


def test_future_keep_policy_and_verdicts() -> None:
    validate_future_keep_policy("OPT-111", load_contract("OPT-111"))
    blocked = decide_verdict(
        primitive={"primitive_win": True},
        parity={"pass": True},
        xor_screen={"xor_screened": False},
        complete={"keep_bar_pass": True, "skipped": False},
        p4096={"faster": True, "within_2pct": True, "skipped": False},
        d128={"within_opt114_2pct": True, "skipped": False},
        d2048={"within_opt114_2pct": True, "skipped": False},
        quality={"model_quality_pass": False, "skipped": True},
        state={"pass": True, "skipped": False},
        mode="acceptance",
    )
    assert blocked["verdict"] == "quality_blocked"
    assert blocked["keep"] is False
    lost = decide_verdict(
        primitive={"phase": "primitive", "primitive_win": False, "skipped": False},
        parity={"pass": True},
        xor_screen=None,
        complete=None,
        p4096=None,
        d128=None,
        d2048=None,
        quality=None,
        state=None,
        mode="acceptance",
    )
    assert lost["verdict"] == "primitive_rejected"
    assert lost["keep"] is False


def test_quality_phase_wires_opt058_and_does_not_stub_keep_nll() -> None:
    quality_src = inspect.getsource(run_quality_phase)
    native_src = inspect.getsource(_run_quality_native)
    assert "QUALITY_NATIVE" in quality_src
    assert "--quality" in native_src
    assert "--quality-config" in native_src
    assert "--attention-pipeline" in native_src
    assert "candidate_nll_not_measured" not in quality_src
    assert "candidate_nll_not_measured" not in native_src
    control_cases = [
        {"name": "held_out_wikitext_1024", "mean_nll": 1.80},
        {"name": "wikitext_nll", "mean_nll": 1.52},
        {"name": "recurrence_short", "mean_nll": 1.77},
        {"name": "recurrence_long", "mean_nll": 1.78},
    ]
    control = {"cases": control_cases}
    candidate = {
        "cases": [
            {"name": "held_out_wikitext_1024", "mean_nll": 1.801},
            {"name": "wikitext_nll", "mean_nll": 1.521},
            {"name": "recurrence_short", "mean_nll": 1.771},
            {"name": "recurrence_long", "mean_nll": 1.781},
        ],
        "new_functional_failures": 0,
        "new_greedy_mismatch": False,
    }
    control_eval = evaluate_measured_quality(CONTROL_ID, measured=control)
    candidate_eval = evaluate_measured_quality(
        BASE_ID, measured=candidate, control_measured=control
    )
    assert control_eval["model_quality_pass"] is True
    assert candidate_eval["model_quality_pass"] is True
    assert candidate_eval["ppl_ratio_held_out"] <= 1.01
    assert ppl_ratio(1.801, 1.80) <= 1.01
    rec_delta = recurrence_incremental_nll(
        {"mean_nll": 1.771},
        {"mean_nll": 1.781},
        1.77,
        1.78,
    )
    assert abs(rec_delta) <= 0.02
    runner = RUNNER.read_text(encoding="utf-8")
    assert "no_performance_survivor" in runner
    assert "--attention-pipeline-control" in runner
    assert "kv_once" in runner
