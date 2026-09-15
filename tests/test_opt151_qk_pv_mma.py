"""Host tests for OPT-151 QK/PV output-producing MMA decode attention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt151_qk_pv_mma import (
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
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    WARMUPS,
    complete_attention_ms,
    eligibility,
    family_plan,
    load_contract,
    pairs_from_graph_ab,
    validate_fixture,
)
from tools.performance_keep_policy import freeze_hash, validate_opt_in_contract
from tools.run_optimization_task import (
    describe_plan,
    load_contract as load_iteration,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE_SRC = ROOT / "cuda/opt151_qk_pv_mma_test.cu"
KERNEL = ROOT / "cuda/opt151_qk_pv_mma_decode.cuh"
PIN = ROOT / "cuda/attention_decode_path.cuh"
ATTN = ROOT / "cuda/attention_decode.cu"
QUALITY_SRC = ROOT / "cuda/opt058_quality_baseline_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
RUNNER = ROOT / "tools/opt151_qk_pv_mma.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt151_qk_pv_mma_provenance.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-151")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    pin = PIN.read_text(encoding="utf-8")
    attn = ATTN.read_text(encoding="utf-8")
    quality = QUALITY_SRC.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert NATIVE_SRC.is_file()
    assert KERNEL.is_file()
    assert contract["task"] == "OPT-151"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["candidate"] == CANDIDATE
    assert contract["candidate_id"] == CANDIDATE
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["dense_bf16_kv"] is True
    assert contract["shadow_f16_cache"] is False
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["target_workloads"] == ["d8192", "d32768"]
    assert "d131040" in contract["guard_workloads"]
    assert iteration["diagnostics_make_target"] == "cuda-opt151-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt151-diagnostics" in makefile
    assert "qw38-cuda-opt151-qk-pv-mma-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "inspect|identity|primitive|screen|same-math|" in native
    assert "independently_restored=true" in native
    assert "same_binary=true" in native
    assert "restore_checkpoint" in native
    assert "prefix_restored" in native
    assert "request_throughput" in native
    assert "QW38_OPT151_QK_PV_MMA_RESULT=" in native
    assert "QW38_OPT151_NATIVE_COUNTS=" in native
    assert "QW38_OPT151_NATIVE_COUNTS=" in task_runner
    assert "QW38_OPT151_QK_PV_MMA_RESULT=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "apply_opt151_qk_pv_mma_ident" in quality
    assert "effective_decode_graph_topology_count" in scheduler
    assert "opt151_qk_pv_mma_decode.cuh" in attn
    assert "decode_attention_qk_pv_mma_v2" in kernel
    assert "__float2half_rn(__bfloat162float" in kernel
    assert "mma_qk_16x8" in kernel
    assert "mma_pv_16x8" in kernel
    assert "0.0F * mma_scores" not in kernel
    assert "kNwarps = 2" in kernel
    assert "mma-dep" in native
    assert contract["qk_mma"] is True
    assert contract["pv_mma"] is True
    assert contract["nthreads"] == 64
    assert "kSelectedOpt151QkPvMma" in pin
    assert "kLegalDecodeAttentionQkPvMmaV2" in pin
    assert PHASES[0] == "preflight"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["license"] == "MIT"
    assert provenance["llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    validate_future_keep_policy("OPT-151", iteration)
    validate_opt_in_contract(contract)


def test_dispatch_buckets_and_frozen_partitions() -> None:
    pin = PIN.read_text(encoding="utf-8")
    contract = load_contract()
    native = NATIVE_SRC.read_text(encoding="utf-8")
    assert "kOpt137MmaThreshold = 8192" in pin
    assert "kOpt137Bucket32768 = 32768" in pin
    assert "kOpt137Bucket65536 = 65536" in pin
    assert "kOpt137TopologyCount = 5" in pin
    assert "kOpt137RepVisible8448" in pin
    assert "kOpt137RepVisible33024" in pin
    assert "kOpt137RepVisible131072" in pin
    assert contract["mma_threshold"] == 8192
    assert contract["representative_visibilities"] == [8448, 33024, 131072]
    assert "8191, 8192, 8193, 32767, 32768, 32769, 65535, 65536, 131071" in native
    assert "128, 1023, 1024, 4096, 4097" in native
    assert "frozen_n_parts_8448" in native
    assert "(42 + index * 997)" in native
    assert "--capacity" in native


def test_keep_policy_roles_and_quality_wiring() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-151")
    roles = {row["workload"]: "target" for row in contract["targets"]}
    roles.update({row["workload"]: "guard" for row in contract["guards"]})
    assert roles["d8192"] == "target"
    assert roles["d32768"] == "target"
    assert roles["d128"] == "guard"
    assert roles["d2048"] == "guard"
    assert roles["p4096"] == "guard"
    assert roles["d131040"] == "guard"
    d131040 = next(row for row in contract["guards"] if row["workload"] == "d131040")
    assert d131040["metrics"] == ["decode_only"]
    assert contract["dispatch_region"] == "decode>=8192"
    assert "decode<8192" in contract["unchanged_branches"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    freeze_hash(contract)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt151-qk-pv-mma")
    assert FIXTURE.name == "opt151_qk_pv_mma.json"


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-151")
    inspect = iteration["workloads"]["preflight"]
    correctness = iteration["workloads"]["correctness"]
    screen = iteration["workloads"]["screen"]
    quality = iteration["workloads"]["quality"]
    performance = iteration["workloads"]["performance"]
    assert loop_product(workload_for_mode(inspect, "feedback")) == 1
    assert loop_product(workload_for_mode(correctness, "feedback")) == 28
    assert loop_product(workload_for_mode(screen, "feedback")) == 16
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(performance, "acceptance")) == 156
    described = describe_plan("OPT-151", "feedback", iteration, "preflight")
    assert "phase=preflight" in described
    assert "loop_product=1" in family_plan("feedback", "preflight")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "target_guard_v2" in proof or "target l>1.00" in proof


def test_missing_long_evidence_and_graph_capacity() -> None:
    empty = pairs_from_graph_ab(
        {}, workload="d8192", prefix=8192, eval_count=256, metric="decode_only"
    )
    assert empty == []
    record = {
        "pairs": [
            {
                "sample_index": 0,
                "order": "AB",
                "A": {"decode_only_tok_s": 10.0, "request_tok_s": 4.0, "p95_ms": 20.0},
                "B": {"decode_only_tok_s": 12.0, "request_tok_s": 4.1, "p95_ms": 19.0},
            }
        ]
    }
    rows = pairs_from_graph_ab(
        record, workload="d8192", prefix=8192, eval_count=256, metric="decode_only"
    )
    assert rows[0]["identity"]["capacity"] == 131072
    assert rows[0]["identity"]["metric_boundary"] == "decode_only"
    assert rows[0]["candidate_rate"] == 12.0
    mismatched = dict(rows[0])
    mismatched["identity"] = {**rows[0]["identity"], "capacity": 4096}
    assert mismatched["identity"]["capacity"] != 131072
    eligible = eligibility()
    assert "coverage_ok" in eligible
    assert eligible["used_opt129_0_113"] is False
    if eligible["coverage_ok"]:
        assert (
            complete_attention_ms(
                json.loads(
                    (
                        ROOT
                        / "evidence/optimization/opt136-graph-accounting/family-gaps.json"
                    ).read_text(encoding="utf-8")
                ),
                engine="quartz",
                prefix=8192,
            )
            > 0.0
        )


def test_pins_match_keep_state() -> None:
    pin = PIN.read_text(encoding="utf-8")
    kept = False
    if FIXTURE.is_file():
        kept = bool(_json(FIXTURE).get("production_kept"))
    if kept:
        assert "kSelectedOpt151QkPvMma = true" in pin
    else:
        assert "kSelectedDecodeAttentionVerifiedMax = 4096" in pin
        assert "kSelectedVec128NParts = 16" in pin
        assert "kSelectedOpt151QkPvMma = false" in pin
    contract = load_contract()
    assert contract["parent_decode_attention"] == PARENT
    assert contract["candidate"] == CANDIDATE


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-151"
    assert payload["parent"] == PARENT_STACK
    assert payload["candidate"] == CANDIDATE
    quality = payload.get("quality") or {}
    if payload.get("mode") == "acceptance" and payload.get("verdict") != "screened_out":
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("candidate_nll_not_measured") is not True
    if payload.get("verdict") == "screened_out":
        assert quality.get("opt058_invoked") is not True
        assert payload.get("production_kept") is False
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["shipping_decode_attention"] == CANDIDATE
    else:
        assert payload["shipping_decode_attention"] == PARENT
    checked = validate_fixture(payload)
    assert checked["ok"] is True
    assert REPORT.is_file() or payload.get("mode") != "acceptance"
