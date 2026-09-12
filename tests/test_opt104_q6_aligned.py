"""Host tests for OPT-104 aligned Q6_K SoA admission. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt104_q6_aligned import (
    ALIGNMENT_BYTES,
    CANDIDATE_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    LAYOUT_VERSION,
    MIN_Q6_SAVING_MS,
    Q6_LAUNCHES,
    REPORT,
    VERDICT_KEYS,
    aligned_layout_desc,
    decide_independent_verdicts,
    dispatch_ok,
    empty_verdict_row,
    evaluate_quality,
    family_plan,
    load_json,
    pack_q6_aligned,
    parity_catalog,
    planned_observation,
    replay_command,
    unpack_q6_aligned,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    validate_performance_admission,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CUDA_PARITY = ROOT / "cuda/opt104_q6_aligned_test.cu"
Q6_PATH = ROOT / "cuda/q6k_decode_path.cuh"
LAYOUT = ROOT / "cuda/q6k_aligned_layout.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-104")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-104"
    assert contract["claims_throughput"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["q6_decode_path_fixed"] == "integer_q8_1"
    assert contract["min_q6_saving_ms"] == MIN_Q6_SAVING_MS
    assert contract["layout_version"] == LAYOUT_VERSION
    assert contract["alignment_bytes"] == ALIGNMENT_BYTES
    assert contract["replacement_mandatory_on_keep"] is True
    assert contract["no_extra_persistent_q6"] is True
    assert contract["quality_contract_id"] == "opt089_strict"
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        "raw_gguf",
        "aligned_soa",
    ]
    assert iteration["task"] == "OPT-104"
    assert iteration["diagnostics_make_target"] == "cuda-opt104-diagnostics"
    assert iteration["target"] == "tools/opt104_q6_aligned.py"
    assert iteration["case_ids"] == [
        "parity",
        "q6",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert "cuda-opt104-diagnostics" in makefile
    assert "qw38-cuda-opt104-q6-aligned-test" in makefile
    assert "qw38-cuda-component-replay" in makefile
    assert "qw38-cuda-optimization-engine-probe" in makefile
    q6 = Q6_PATH.read_text(encoding="utf-8")
    layout = LAYOUT.read_text(encoding="utf-8")
    fixture = load_json(FIXTURE)
    shipping = str(fixture.get("shipping_layout") or "raw_gguf")
    assert 'kSelectedQ6DeviceLayout[] = "' in q6
    assert f'kSelectedQ6DeviceLayout[] = "{shipping}"' in q6
    assert 'kSelectedQ6DecodePath[] = "integer_q8_1"' in q6
    assert "kQ6KAlignedLayoutVersion = 1" in layout
    assert "kQ6KAlignedAlignBytes = 64" in layout
    assert "QW38_OPT104_RESULT=" in CUDA_PARITY.read_text(encoding="utf-8")
    assert "QW38_OPT104_CASE=" in CUDA_PARITY.read_text(encoding="utf-8")


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-104")
    parity = iteration["workloads"]["parity"]
    q6 = iteration["workloads"]["q6"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 12
    assert loop_product(workload_for_mode(q6, "feedback")) == 8
    assert loop_product(workload_for_mode(q6, "acceptance")) == 26
    assert loop_product(workload_for_mode(quality, "release")) == 12
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    assert family_plan("parity", "feedback")["cases"] == 12
    screen = family_plan("q6", "feedback")
    assert screen["candidates"] == 2
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    assert screen["engine_pairs"] == 1
    accept = family_plan("q6", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    assert accept["engine_pairs"] == 5
    described = describe_plan("OPT-104", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "replacement mandatory" in proof


def test_parity_catalog_shapes() -> None:
    cases = parity_catalog()
    assert len(cases) == 12
    ids = {row["id"] for row in cases}
    assert "Q6_aligned_M1_N1_K256_inverse_mmv" in ids
    assert "Q6_aligned_M3_N1_K256_pad_inverse_mmv" in ids
    assert "Q6_aligned_attn_out_M5120_K6144" in ids
    assert "Q6_aligned_logits_geom_byte_neutral" in ids
    assert "Q6_aligned_integer_override_M17_K5120" in ids
    assert "Q6_aligned_empty_ok" in ids


def test_two_layout_configs_q6_fixed() -> None:
    assert [row["id"] for row in CONFIGS] == ["raw_gguf", "aligned_soa"]
    assert all(row["q6_decode"] == "integer_q8_1" for row in CONFIGS)
    assert CONFIGS[0]["q6_device_layout"] == "raw_gguf"
    assert CONFIGS[1]["q6_device_layout"] == "aligned_soa"
    assert all(row["expected_q6_launches"] == Q6_LAUNCHES for row in CONFIGS)


def test_layout_pack_unpack_is_byte_exact() -> None:
    rows, columns = 3, 256
    desc = aligned_layout_desc(rows, columns)
    assert desc["d_pad_bytes"] == 58
    assert desc["byte_neutral"] is False
    gguf = bytes((index * 73 + 19) & 0xFF for index in range(desc["gguf_bytes"]))
    soa = pack_q6_aligned(gguf, rows, columns)
    back = unpack_q6_aligned(soa, rows, columns)
    assert back == gguf
    attn = aligned_layout_desc(5120, 6144)
    assert attn["byte_neutral"] is True
    assert attn["total_bytes"] == attn["gguf_bytes"]
    logits = aligned_layout_desc(248320, 5120)
    assert logits["byte_neutral"] is True


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-104")
    validate_future_keep_policy("OPT-104", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-104",
            {
                "task": "OPT-104",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_independent_verdicts_keep_aligned_when_all_pass() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            CANDIDATE_ID: {"performance_pass": True, "mean_diff_ms": 0.12},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == CANDIDATE_ID
    assert decided["production_kept"] is True
    assert decided["shipping_layout"] == CANDIDATE_ID
    assert decided["independent_verdicts"][CANDIDATE_ID]["production_kept"] is True


def test_independent_verdicts_retain_raw_without_perf() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            CANDIDATE_ID: {"performance_pass": False, "mean_diff_ms": 0.02},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == CONTROL_ID
    assert decided["production_kept"] is False
    assert decided["shipping_layout"] == CONTROL_ID


def test_feedback_cannot_keep() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            CANDIDATE_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            CANDIDATE_ID: {"performance_pass": True, "mean_diff_ms": 0.12},
        },
        mode="feedback",
    )
    assert decided["production_kept"] is False
    assert all(
        not row["production_kept"] for row in decided["independent_verdicts"].values()
    )


def test_dispatch_ok_checks_layout_and_q6_launches() -> None:
    raw = "mixer_calls=17 attention_input_output_groups=16 q6_device_layout=raw_gguf\n"
    aligned = (
        "mixer_calls=17 attention_input_output_groups=16 q6_device_layout=aligned_soa\n"
    )
    assert dispatch_ok(CONFIGS[0], raw) is True
    assert dispatch_ok(CONFIGS[1], aligned) is True
    assert dispatch_ok(CONFIGS[1], raw) is False


def test_replay_command_uses_device_layout() -> None:
    command = replay_command(
        {"warmups": 1, "samples": 3},
        CONFIGS[1],
        None,
    )
    assert "--workload" in command
    assert command[command.index("--workload") + 1] == "decode-q6"
    assert "--q6-device-layout" in command
    assert command[command.index("--q6-device-layout") + 1] == "aligned_soa"


def test_quality_reuses_opt089_authenticated_scores() -> None:
    control = evaluate_quality(CONFIGS[0])
    candidate = evaluate_quality(CONFIGS[1])
    assert control["reuse_opt089_authenticated_scores"] is False
    assert candidate["reuse_opt089_authenticated_scores"] is True
    assert "opt089_admission" in candidate
    assert candidate["quality_contract_id"] == "opt089_strict"


def test_screen_keep_is_rejected() -> None:
    iteration = load_contract("OPT-104")
    q6 = workload_for_mode(iteration["workloads"]["q6"], "feedback")
    stdout = json.dumps(
        {
            "warmups": 1,
            "samples": 3,
            "observed_warmups": 1,
            "observed_samples": 3,
            "observed_candidates": 2,
            "observed_shapes": 1,
            "observed_tier": "screen",
            "pairs": 1,
            "sample_ids": [0, 1, 2],
            "acceptance_executed": False,
            "keep": True,
        }
    )
    screen_keep = validate_performance_admission(
        iteration,
        mode="feedback",
        workload_name="q6",
        workload=q6,
        stdout="QW38_OPT104_RESULT=" + stdout,
        success=True,
    )
    assert screen_keep["ok"] is False
    observed = parse_native_observation("QW38_OPT104_RESULT=" + stdout)
    assert observed["keep"] is True


def test_parse_native_observation_opt104_prefix() -> None:
    payload = (
        'QW38_OPT104_RESULT={"task":"OPT-104","phase":"parity","success":true}\n'
        'QW38_OPT104_NATIVE_COUNTS={"observed_shapes":12,'
        '"observed_tier":"correctness"}\n'
    )
    observed = parse_native_observation(payload)
    assert observed["task"] == "OPT-104"
    assert observed["phase"] == "parity"
    assert observed["observed_shapes"] == 12


def test_host_phases_write_fixture_and_retain_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt104_q6_aligned as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt104_q6_aligned.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    tmp = tmp_path / "run"
    parity = mod.run("feedback", "parity", tmp, skip_gpu=True)
    assert parity["task"] == "OPT-104"
    assert (tmp_path / "opt104_q6_aligned.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()
    quality = mod.run("release", "quality", tmp, skip_gpu=True)
    assert (quality.get("quality") or {}).get("phase") == "quality"
    q6 = mod.run("feedback", "q6", tmp, skip_gpu=True)
    assert q6["production_kept"] is False
    assert q6["claims_throughput"] is False
    fixture = load_json(tmp_path / "opt104_q6_aligned.json")
    assert fixture["control_layout"] == "raw_gguf"
    assert fixture["shipping_layout"] == "raw_gguf"
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    assert empty_verdict_row()["opt074_coverage_unadmitted_blocker"] is False


def test_committed_fixture_skeleton() -> None:
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    q6 = Q6_PATH.read_text(encoding="utf-8")
    contract = _json(CONTRACT)
    for key in contract["required_fixture_keys"]:
        assert key in fixture
    assert fixture["task"] == "OPT-104"
    assert fixture["control_layout"] == "raw_gguf"
    assert fixture["shipping_layout"] in {"raw_gguf", "aligned_soa"}
    assert fixture["production_kept"] == (fixture["shipping_layout"] == "aligned_soa")
    assert fixture["claims_throughput"] == (
        fixture["production_kept"] and fixture.get("evidence_complete")
    )
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert "raw_gguf" in fixture["independent_verdicts"]
    assert "aligned_soa" in fixture["independent_verdicts"]
    kept = [
        cid
        for cid, row in fixture["independent_verdicts"].items()
        if row["production_kept"]
    ]
    assert len(kept) <= 1
    if fixture["production_kept"]:
        assert kept == ["aligned_soa"]
    assert f'kSelectedQ6DeviceLayout[] = "{fixture["shipping_layout"]}"' in q6
    assert "kernel_parity_pass" in report
    assert "opt074_coverage_unadmitted" in report.casefold()


def test_planned_observation_counts() -> None:
    plan = family_plan("q6", "feedback")
    observed = planned_observation(plan)
    assert observed["task"] == "OPT-104"
    assert observed["observed_candidates"] == 2
    assert observed["observed_samples"] == 3
    assert observed["keep"] is False
