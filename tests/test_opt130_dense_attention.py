"""Host tests for OPT-130 dense decode-attention occupancy/vec128_open."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt130_dense_attention import (
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
    validate_fixture,
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
NATIVE_SRC = ROOT / "cuda/opt130_dense_attention_test.cu"
PIN = ROOT / "cuda/attention_decode_path.cuh"
ATTN = ROOT / "cuda/attention_decode.cu"
QUALITY_SRC = ROOT / "cuda/opt058_quality_baseline_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt130_dense_attention_provenance.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-130")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    pin = PIN.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    quality = QUALITY_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-130"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["candidate"] == CANDIDATE
    assert contract["candidate_id"] == "occupancy_partition_or_gqa_kv_reuse"
    assert contract["second_candidate_launched"] is False
    assert contract["gqa6_admitted"] is False
    assert contract["mma_admitted"] is False
    assert contract["require_candidate_nll"] is True
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["metrics"] == ["decode_only", "complete_request"]
    assert contract["dense_bf16_kv"] is True
    assert contract["packed_kv_admitted"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt130-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt130-diagnostics" in makefile
    assert "qw38-cuda-opt130-dense-attention-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "inspect|identity|primitive|same-math|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "decode_only_tok_s" in native
    assert "request_tok_s" in native
    assert "QW38_OPT130_DENSE_ATTENTION_RESULT=" in native
    assert "QW38_OPT130_NATIVE_COUNTS=" in native
    assert "QW38_OPT130_NATIVE_COUNTS=" in runner
    assert "QW38_OPT130_DENSE_ATTENTION_RESULT=" in runner
    assert "occupancy_snapped_vec128_n_parts" in attn
    assert "kOpt130CandidateVec128NParts = 8" in pin
    assert "kSelectedDecodeAttentionVerifiedMax = 4096" in pin
    assert "kSelectedVec128NParts = 16" in pin
    assert "apply_opt130_dense_attention_ident" in pin
    assert "apply_opt130_dense_attention_ident" in quality
    assert "vec128_n_parts" in quality
    assert PHASES[0] == "inspect"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["license"] == "MIT"
    validate_future_keep_policy("OPT-130", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-130")
    inspect = iteration["workloads"]["inspect"]
    identity = iteration["workloads"]["identity"]
    primitive = iteration["workloads"]["primitive"]
    same = iteration["workloads"]["same-math"]
    cancel = iteration["workloads"]["cancellation"]
    screen = iteration["workloads"]["screen-d128"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(inspect, "feedback")) == 2
    assert loop_product(workload_for_mode(identity, "feedback")) == 12
    assert loop_product(workload_for_mode(primitive, "feedback")) == 96
    assert loop_product(workload_for_mode(same, "feedback")) == 16
    assert loop_product(workload_for_mode(cancel, "feedback")) == 2
    assert loop_product(workload_for_mode(screen, "feedback")) == 16
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    described = describe_plan("OPT-130", "feedback", iteration, "inspect")
    assert "phase=inspect" in described
    assert "loop_product=2" in family_plan("feedback", "inspect")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "decode-only and complete-request" in proof
    assert "opt-058" in proof


def test_keep_requires_opt125_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-130")
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
    assert REPORT.parent.as_posix().endswith("opt130-dense-attention")
    assert FIXTURE.name == "opt130_dense_attention.json"


def test_pins_match_keep_state() -> None:
    pin = PIN.read_text(encoding="utf-8")
    kept = False
    if FIXTURE.is_file():
        kept = bool(_json(FIXTURE).get("production_kept"))
    if kept:
        assert "kSelectedDecodeAttentionVerifiedMax = 131072" in pin
        assert "kSelectedVec128NParts = 8" in pin
    else:
        assert "kSelectedDecodeAttentionVerifiedMax = 4096" in pin
        assert "kSelectedVec128NParts = 16" in pin
    contract = _json(CONTRACT)
    assert contract["parent_decode_attention"] == PARENT
    assert contract["candidate"] == CANDIDATE


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-130"
    assert payload["parent"] == PARENT_STACK
    assert payload["candidate"] == CANDIDATE
    quality = payload.get("quality") or {}
    if payload.get("mode") == "acceptance":
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["shipping_decode_attention"] == CANDIDATE
    else:
        assert payload["shipping_decode_attention"] == PARENT
    checked = validate_fixture(payload)
    assert checked["ok"] is True
    assert REPORT.is_file() or payload.get("mode") != "acceptance"
