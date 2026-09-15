"""Host tests for OPT-140 production-boundary prefill attention replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt138_remaining_gap_profile import replay_family_for
from tools.opt139_counter_identity import (
    LAUNCH_PREFILL_ATTN,
    OPT140_INELIGIBLE,
    identity_match,
    kernel_stem,
    production_prefill_attn_identity,
)
from tools.opt140_prefill_attention_replay import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    boundary_manifest,
    family_plan,
    parse_layer_parity,
    parse_parity_summary,
    parse_rounds,
    validate_fixture,
)
from tools.performance_evidence import replay_production_boundary_ok
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
REPLAY_SRC = ROOT / "cuda/optimization_component_replay.cu"
REPLAY_HDR = ROOT / "cuda/optimization_component_replay.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
OPT138_TOOL = ROOT / "tools/opt138_remaining_gap_profile.py"
OPT111 = ROOT / "cuda/opt111_llama_prompt_attention.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-140")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    replay = REPLAY_SRC.read_text(encoding="utf-8")
    header = REPLAY_HDR.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    opt138 = OPT138_TOOL.read_text(encoding="utf-8")
    opt111 = OPT111.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt140_prefill_attention_replay.py").read_text(
        encoding="utf-8"
    )
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-140"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["prefill_tokens"] == 4096
    assert contract["capacity"] == 131072
    assert contract["decode_attention_substitution"] is False
    assert contract["expected_kernel"] == LAUNCH_PREFILL_ATTN
    assert contract["new_quality_tolerance"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt140-diagnostics"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt140-diagnostics" in makefile
    assert "QW38_OPT140_PREFILL_ATTENTION_REPLAY_RESULT=" in runner
    assert "kPromptAttention" in header
    assert "prompt-attention" in replay
    assert "--opt140-protocol" in replay
    assert "maybe_capture_prompt_attention_inputs" in scheduler
    assert "replay_family_for" in opt138
    assert "fattn_mma_pipeline_opt111_base" in opt111
    assert "decode_attention_substitution" in tool
    assert PHASES == ("preflight", "replay", "counters", "report")
    validate_future_keep_policy("OPT-140", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-140")
    preflight = iteration["workloads"]["preflight"]
    replay = iteration["workloads"]["replay"]
    counters = iteration["workloads"]["counters"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(replay, "acceptance")) == 4
    assert loop_product(workload_for_mode(counters, "acceptance")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-140", "acceptance", iteration, "replay")
    assert "phase=replay" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def test_phase_specific_replay_family() -> None:
    assert replay_family_for("prefill", "attn_core") == "prompt-attention"
    assert replay_family_for("prefill", "attn_fused") == "prompt-attention"
    assert replay_family_for("decode", "attn_core") == "decode-attention"
    assert replay_family_for("prefill", "prompt_mmq") == "prompt-ffn"
    ok = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "prefill",
            "replay_family": "prompt-attention",
        }
    )
    assert ok["ok"] is True
    mismatch = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "prefill",
            "replay_family": "decode-attention",
        }
    )
    assert mismatch["ok"] is False
    assert mismatch["reason"] == "replay/production boundary mismatch"


def test_opt139_prefill_identity_and_decode_rejection() -> None:
    expected = production_prefill_attn_identity()
    assert expected["expected_kernel"] == LAUNCH_PREFILL_ATTN
    assert expected["replay_family"] == "prompt-attention"
    assert expected["capacity"] == 131072
    matched = identity_match(
        phase="prefill",
        engine="quartz",
        workload="P4096",
        kernel=LAUNCH_PREFILL_ATTN,
        expected_kernel=expected["expected_kernel"],
        replay_family="prompt-attention",
        replay_boundary=None,
    )
    assert matched["ok"] is True
    rejected = identity_match(
        phase="prefill",
        engine="quartz",
        workload="P4096",
        kernel=LAUNCH_PREFILL_ATTN,
        expected_kernel=LAUNCH_PREFILL_ATTN,
        replay_family="decode-attention",
        replay_boundary=None,
    )
    assert rejected["ok"] is False
    assert OPT140_INELIGIBLE in rejected["mismatches"]
    mangled = (
        "void qw38::cuda::fattn_mma_pipeline_opt111_base<16, (bool)0, "
        "(bool)1>(qw38::cuda::AttentionConfig)"
    )
    assert kernel_stem(mangled) == LAUNCH_PREFILL_ATTN
    assert kernel_stem("fattn_mma_pipeline_kernel") == LAUNCH_PREFILL_ATTN
    manifest = boundary_manifest(
        replay_family="decode-attention",
        dispatch=None,
        identity=rejected,
    )
    assert manifest["decode_attention_substitution"] is True
    assert manifest["reason"] == OPT140_INELIGIBLE


def test_parity_and_round_parsers() -> None:
    text = (
        "prompt_attention_parity layer=0 slot=0 max_abs=1e-4 kv_max_abs=0 "
        "nonfinite=0 causal_tail_abs=2e-5 chunk_start_abs=1e-5 "
        "chunk_mid_abs=3e-5 equal=true\n"
        "prompt_attention_parity_summary layers=16 max_abs=0.0002 "
        "kv_max_abs=0.0001 nonfinite=0 equal=true existing_abs_tol=0.002 "
        "new_quality_tolerance=false\n"
        "round family=prompt-attention cache_mode=rotating warmup=true "
        "sample_index=0 observation_unit=independent_round enclosing_ms=12.5 "
        "kernel_only_ms=11.0 gate_up_calls=0\n"
        "round family=prompt-attention cache_mode=rotating warmup=false "
        "sample_index=0 observation_unit=independent_round enclosing_ms=12.1 "
        "kernel_only_ms=10.8 gate_up_calls=0\n"
        "round family=prompt-attention cache_mode=rotating warmup=false "
        "sample_index=1 observation_unit=independent_round enclosing_ms=12.0 "
        "kernel_only_ms=10.7 gate_up_calls=0\n"
        "round family=prompt-attention cache_mode=rotating warmup=false "
        "sample_index=2 observation_unit=independent_round enclosing_ms=12.2 "
        "kernel_only_ms=10.9 gate_up_calls=0\n"
    )
    summary = parse_parity_summary(text)
    assert summary is not None
    assert summary["equal"] is True
    assert summary["layers"] == 16
    layers = parse_layer_parity(text)
    assert layers[0]["causal_tail_abs"] == 2e-5
    rounds = parse_rounds(text)
    assert sum(1 for row in rounds if row["warmup"]) == 1
    assert sum(1 for row in rounds if not row["warmup"]) == 3


def test_fixture_keys() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert fixture["decode_attention_substitution"] is False
    assert REPORT.parent.as_posix().endswith("opt140-prefill-attention-replay")
