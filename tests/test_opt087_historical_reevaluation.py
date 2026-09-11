"""Host tests for OPT-087 historical leftover reevaluation. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt087_historical_reevaluation import (
    CANDIDATE_COUNT,
    CONTRACT,
    DISPOSITIONS,
    FIXTURE,
    ITERATION,
    NO_REPEAT_TASKS,
    OPT072_FIXTURE,
    OPT072_REPORT,
    REPORT,
    REVIEWED_CANDIDATES,
    confirm_candidate,
    leftover_qualifies,
    load_json,
    measured_upside,
    scan_opt072_leftovers,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_host_only_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-087")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-087"
    assert contract["claims_throughput"] is False
    assert contract["no_additional_reopen"] is True
    assert contract["opt074_coverage_unadmitted_is_blocker"] is False
    assert contract["opt074_family_admission_required"] is False
    assert contract["candidate_count"] == CANDIDATE_COUNT
    assert contract["dispositions"] == list(DISPOSITIONS)
    assert iteration["task"] == "OPT-087"
    assert iteration["host_only"] is True
    assert iteration["skip_gpu_setup"] is True
    assert iteration["skip_compile"] is True
    assert iteration["diagnostics_make_target"] == "cuda-opt087-diagnostics"
    assert iteration["case_ids"] == ["inventory", "review"]
    assert iteration["reopen_phases_armed"] is False
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["modes"]["feedback"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 300
    assert "cuda-opt087-diagnostics" in makefile
    assert "cuda-opt087-diagnostics:" in makefile


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-087")
    inventory = iteration["workloads"]["inventory"]
    review = iteration["workloads"]["review"]
    assert inventory["cases"] == CANDIDATE_COUNT
    assert review["cases"] == CANDIDATE_COUNT
    assert inventory["gpu_work"] is False
    assert review["gpu_work"] is False
    assert loop_product(workload_for_mode(inventory, "feedback")) == CANDIDATE_COUNT
    assert loop_product(workload_for_mode(review, "acceptance")) == CANDIDATE_COUNT
    plan = describe_plan("OPT-087", "feedback", iteration, "inventory")
    assert "phase=inventory" in plan
    assert "host_only=true" in plan
    assert "historical_oracles=none" in plan
    assert f"product={CANDIDATE_COUNT}" in plan
    review_plan = describe_plan("OPT-087", "acceptance", iteration, "review")
    assert "phase=review" in review_plan
    assert "host_only=true" in review_plan
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "historical opt-072 report unmodified" in proof
    assert "opt074_coverage_unadmitted is not a blocker" in proof
    assert "do not rerun opt-077/078" in proof


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    validate_future_keep_policy("OPT-087", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-087",
            {
                "task": "OPT-087",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_reviewed_table_matches_dossier_and_has_evidence() -> None:
    ids = [row["id"] for row in REVIEWED_CANDIDATES]
    assert len(ids) == CANDIDATE_COUNT == 20
    assert len(ids) == len(set(ids))
    assert "OPT-075:q4_integer" in ids
    assert "OPT-076:q4_late" in ids
    assert "OPT-064:q8_layout" in ids
    assert "OPT-066:mmq_x_pipeline" in ids
    assert "OPT-077:tiled_decode_gdn" in ids
    assert "OPT-078:prepared_q_veckv" in ids
    assert "OPT-079:kv_once" in ids
    assert "half-scale-q8-1" in ids
    assert "OPT-042:cud001_one_warp" in ids
    for task in NO_REPEAT_TASKS:
        assert any(row.get("opt072_task") == task for row in REVIEWED_CANDIDATES)
    for row in REVIEWED_CANDIDATES:
        assert row["disposition"] in DISPOSITIONS
        assert row["disposition"] != "reopen"
        assert (ROOT / row["evidence_path"]).is_file()
        if row.get("historical_evidence"):
            assert (ROOT / row["historical_evidence"]).is_file()


def test_opt072_leftover_scan_finds_no_reopen() -> None:
    register = load_json(OPT072_FIXTURE)
    leftovers = scan_opt072_leftovers(register)
    assert leftovers == []
    owned = next(
        row for row in register["entries"] if row["id"] == "OPT-046:integer_q8_w4"
    )
    assert measured_upside(owned) is True
    assert leftover_qualifies(owned, covered=set()) is False
    half = next(
        row
        for row in register["entries"]
        if row["id"] == "OPT-046:integer_q8_1_w4_half_scale"
    )
    assert leftover_qualifies(half, covered=set()) is False
    loser = next(row for row in register["entries"] if row["task"] == "OPT-068")
    assert leftover_qualifies(loser, covered=set()) is False
    nan = next(
        row
        for row in register["entries"]
        if row["id"] == "OPT-046:historical_scheduler_nan"
    )
    assert leftover_qualifies(nan, covered=set()) is False


def test_synthetic_numerical_leftover_can_qualify() -> None:
    entry = {
        "id": "OPT-999:synthetic",
        "kind": "missing_admission",
        "disposition": "numerically_eligible",
        "next_owner": "none",
        "latest_equivalent_path": "implemented diagnostic",
        "v2_comparison": {
            "current_complete_cost_ms": {"packed": 10.0, "candidate": 7.0}
        },
    }
    assert leftover_qualifies(entry, covered=set()) is True
    covered = leftover_qualifies(entry, covered={"OPT-999:synthetic"})
    assert covered is False
    entry["next_owner"] = "OPT-085"
    assert leftover_qualifies(entry, covered=set()) is False


def test_pre_documented_confirmers_against_historical_files() -> None:
    register = load_json(OPT072_FIXTURE)
    for spec in REVIEWED_CANDIDATES:
        confirmed = confirm_candidate(spec, register)
        assert confirmed["id"] == spec["id"]
        assert confirmed["disposition"] == spec["disposition"]
        assert confirmed["reopen"] is False
        assert confirmed["confirmation"]["confirmed"] is True


def test_historical_opt072_report_unmodified() -> None:
    assert OPT072_FIXTURE.is_file()
    assert OPT072_REPORT.is_file()
    fixture = load_json(OPT072_FIXTURE)
    text = OPT072_REPORT.read_text(encoding="utf-8")
    assert fixture["task"] == "OPT-072"
    assert fixture["status"] == "rejection_review"
    assert "OPT-072" in text
    assert fixture["claims_throughput"] is False


def test_host_phases_write_fixture_and_no_additional_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt087_historical_reevaluation as mod

    monkeypatch.setattr(
        mod, "FIXTURE", tmp_path / "opt087_historical_reevaluation.json"
    )
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    monkeypatch.setattr(mod, "EVIDENCE", tmp_path)
    inventory = mod.run("feedback", "inventory", tmp_path / "run")
    assert inventory["task"] == "OPT-087"
    assert inventory["no_additional_reopen"] is True
    assert inventory["reopened_candidate"] is None
    assert inventory["claims_throughput"] is False
    assert inventory["opt074_coverage_unadmitted_blocker"] is False
    assert inventory["production_pins_unchanged"] is True
    assert inventory["historical_opt072_unmodified"] is True
    assert len(inventory["candidates"]) == CANDIDATE_COUNT
    for row in inventory["candidates"]:
        assert row["disposition"] in DISPOSITIONS
        assert row["disposition"] != "reopen"
        assert row["evidence_path"]
    review = mod.run("acceptance", "review", tmp_path / "run")
    assert review["no_additional_reopen"] is True
    assert review["mode"] == "acceptance"
    dumped = load_json(tmp_path / "opt087_historical_reevaluation.json")
    assert dumped["no_additional_reopen"] is True
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "no_additional_reopen" in report
    lowered = report.casefold()
    assert "kernel parity" in lowered or "kernel_parity" in lowered
    assert "not" in lowered and "blocker" in lowered
    assert "OPT-077" in report
    assert "OPT-078" in report
    assert "half-scale" in lowered


def test_committed_fixture_schema() -> None:
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    fixture = load_json(FIXTURE)
    report = REPORT.read_text(encoding="utf-8")
    assert fixture["task"] == "OPT-087"
    assert fixture["no_additional_reopen"] is True
    assert fixture["reopened_candidate"] is None
    assert fixture["claims_throughput"] is False
    assert fixture["opt074_coverage_unadmitted_blocker"] is False
    assert fixture["historical_opt072_unmodified"] is True
    assert fixture["production_pins_unchanged"] is True
    assert len(fixture["candidates"]) == CANDIDATE_COUNT
    for row in fixture["candidates"]:
        assert row["disposition"] in {"reopen", "owned_elsewhere", "do_not_reopen"}
        assert "evidence_path" in row
        assert (ROOT / row["evidence_path"]).is_file()
    kept_reopen = [
        row for row in fixture["candidates"] if row["disposition"] == "reopen"
    ]
    assert kept_reopen == []
    assert "no_additional_reopen" in report
    assert "opt074" in report.casefold()
    assert "not" in report.casefold() and "blocker" in report.casefold()
    stdout = "QW38_OPT087_RESULT=" + json.dumps(
        {
            "warmups": 0,
            "samples": 1,
            "observed_warmups": 0,
            "observed_samples": 1,
            "observed_candidates": 1,
            "observed_shapes": CANDIDATE_COUNT,
            "observed_tier": "acceptance",
            "pairs": 1,
            "sample_ids": [0],
            "acceptance_executed": True,
            "keep": False,
            "no_additional_reopen": True,
        }
    )
    observed = parse_native_observation(stdout)
    assert observed["keep"] is False
    assert observed["no_additional_reopen"] is True
