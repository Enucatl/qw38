"""OPT-087 selective reopen of remaining numerical-policy leftovers.

Host-only inventory/review unless a candidate is actually reopened. Confirms
the pre-documented OPT-072 leftover dispositions, scans for at most one
already-implemented candidate with measured upside that was blocked primarily
by numerical policy, and otherwise delivers no_additional_reopen=true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kernel_parity import OPT074_FAMILY_ADMISSION_REQUIRED  # noqa: E402
from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    dump_json,
    load_json,
    utc_now,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt087_historical_reevaluation_contract.json"
ITERATION = ROOT / "pins/opt087_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt087_historical_reevaluation.json"
REPORT = ROOT / "evidence/optimization/opt087-historical-reevaluation/REPORT.md"
EVIDENCE = REPORT.parent
OPT072_FIXTURE = ROOT / "fixtures/opt072_rejection_review.json"
OPT072_REPORT = ROOT / "evidence/optimization/opt072-rejection-review/REPORT.md"
PHASES = ("inventory", "review")
DISPOSITIONS = ("reopen", "owned_elsewhere", "do_not_reopen")
HARD_ERROR_KINDS = frozenset({"layout_nonfinite_state", "stale_object_selector"})
ARITHMETIC_KINDS = frozenset(
    {"numeric_rejection", "missing_admission", "exact_scalar_association"}
)
OWNED_ELSEWHERE_OWNERS = frozenset(
    {"OPT-075", "OPT-076", "OPT-085", "OPT-064", "OPT-066", "OPT-070", "OPT-086"}
)
NO_REPEAT_TASKS = (
    "OPT-067",
    "OPT-065",
    "OPT-068",
    "OPT-024",
    "OPT-027",
    "OPT-028",
    "OPT-029",
    "OPT-030",
    "OPT-037",
    "OPT-054",
    "OPT-055",
)

REVIEWED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "OPT-075:q4_integer",
        "label": "OPT-075/076 Q4 integer / late",
        "disposition": "owned_elsewhere",
        "owner": "OPT-085",
        "reason": "owned by OPT-085; not reopened here",
        "evidence_path": "fixtures/opt085_q4_reevaluation.json",
        "historical_evidence": "fixtures/opt075_q4_production_admission.json",
        "opt072_ids": [
            "OPT-046:integer_q8_w4",
            "OPT-062:integer_q8_complete_ffn",
            "OPT-063:paired_integer",
        ],
    },
    {
        "id": "OPT-076:q4_late",
        "label": "OPT-076 Q4 late reduction",
        "disposition": "owned_elsewhere",
        "owner": "OPT-085",
        "reason": "owned by OPT-085; not reopened here",
        "evidence_path": "fixtures/opt085_q4_reevaluation.json",
        "historical_evidence": "fixtures/opt076_q4_reduction.json",
        "opt072_ids": [],
    },
    {
        "id": "OPT-064:q8_layout",
        "label": "OPT-064/066 Q8/MMQ",
        "disposition": "owned_elsewhere",
        "owner": "OPT-086",
        "reason": "owned by OPT-086; not reopened here",
        "evidence_path": "fixtures/opt086_q8_mmq_reevaluation.json",
        "historical_evidence": "fixtures/opt070_keep_revalidation.json",
        "opt072_ids": ["OPT-064:r2_w2_keep_evidence"],
    },
    {
        "id": "OPT-066:mmq_x_pipeline",
        "label": "OPT-066 MMQ fma_async_x",
        "disposition": "owned_elsewhere",
        "owner": "OPT-086",
        "reason": "owned by OPT-086; not reopened here",
        "evidence_path": "fixtures/opt086_q8_mmq_reevaluation.json",
        "historical_evidence": "fixtures/opt070_keep_revalidation.json",
        "opt072_ids": ["OPT-066:fma_async_x_keep_evidence"],
    },
    {
        "id": "OPT-077:tiled_decode_gdn",
        "label": "OPT-077 tiled decode GDN",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "component_interval_not_positive; performance, not numerical-policy",
        "evidence_path": "fixtures/opt077_gdn_decode.json",
        "historical_evidence": "evidence/optimization/opt077-gdn-decode/REPORT.md",
        "opt072_ids": [],
    },
    {
        "id": "OPT-078:prepared_q_veckv",
        "label": "OPT-078 prepared_q / veckv",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "performance_rejected / negative complete saving",
        "evidence_path": "fixtures/opt078_decode_attention.json",
        "historical_evidence": "evidence/optimization/opt078-decode-attention/REPORT.md",
        "opt072_ids": [],
    },
    {
        "id": "OPT-079:kv_once",
        "label": "OPT-079 kv_once",
        "disposition": "do_not_reopen",
        "owner": "OPT-088",
        "reason": "keep; OPT-088 interaction check only",
        "evidence_path": "fixtures/opt079_attention_kv_operands.json",
        "historical_evidence": "evidence/optimization/opt079-attention-kv-operands/REPORT.md",
        "opt072_ids": [],
        "status_note": "keep",
    },
    {
        "id": "OPT-067:prompt_pair",
        "label": "OPT-067 prompt gate/up pairing",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt067-prompt-pair/REPORT.md",
        "opt072_ids": ["OPT-067:prompt gate/up pairing"],
        "opt072_task": "OPT-067",
    },
    {
        "id": "OPT-065:mmq_tiles",
        "label": "OPT-065 MMQ I/J retune",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt065-mmq-tiles/REPORT.md",
        "opt072_ids": ["OPT-065:MMQ I/J retune with FMA/async Y"],
        "opt072_task": "OPT-065",
    },
    {
        "id": "OPT-068:scoped_codegen",
        "label": "OPT-068 scoped O3/FMA codegen",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt068-scoped-codegen/REPORT.md",
        "opt072_ids": ["OPT-068:scoped O3/FMA codegen"],
        "opt072_task": "OPT-068",
    },
    {
        "id": "OPT-024:mixer_q8_d2r",
        "label": "OPT-024 Blackwell Q8 D2R",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md",
        "opt072_ids": ["OPT-024:Blackwell Q8 D2R"],
        "opt072_task": "OPT-024",
    },
    {
        "id": "OPT-027:persistent_fattn",
        "label": "OPT-027 persistent Ada+ fattn stream-K",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt027-persistent-fattn/REJECTION.md",
        "opt072_ids": ["OPT-027:persistent Ada+ fattn stream-K"],
        "opt072_task": "OPT-027",
    },
    {
        "id": "OPT-028:mmq_streamk",
        "label": "OPT-028 Q4/Q6 MMQ stream-K",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt028-mmq-streamk/REJECTION.md",
        "opt072_ids": ["OPT-028:Q4/Q6 MMQ stream-K"],
        "opt072_task": "OPT-028",
    },
    {
        "id": "OPT-029:gdn_fuse",
        "label": "OPT-029 fused GDN conv/gated output",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt029-gdn-fuse/REJECTION.md",
        "opt072_ids": ["OPT-029:fused GDN conv/gated output"],
        "opt072_task": "OPT-029",
    },
    {
        "id": "OPT-030:pdl_launches",
        "label": "OPT-030 PDL prompt launches",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt030-pdl-launches/REJECTION.md",
        "opt072_ids": ["OPT-030:PDL prompt launches"],
        "opt072_task": "OPT-030",
    },
    {
        "id": "OPT-037:ffn_tiles",
        "label": "OPT-037 FFN I/J tile sweep",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt037-ffn-tiles/REJECTION.md",
        "opt072_ids": ["OPT-037:FFN I/J tile sweep"],
        "opt072_task": "OPT-037",
    },
    {
        "id": "OPT-054:prefill_microbatch",
        "label": "OPT-054 prefill microbatch 512/1024/2048",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt054-prefill-microbatch/REPORT.md",
        "opt072_ids": ["OPT-054:prefill microbatch 512/1024/2048"],
        "opt072_task": "OPT-054",
    },
    {
        "id": "OPT-055:execution_graphs",
        "label": "OPT-055 broader execution graphs",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": "OPT-072 no_repeat performance loser",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt055-execution-graphs/REPORT.md",
        "opt072_ids": ["OPT-055:broader execution graphs"],
        "opt072_task": "OPT-055",
    },
    {
        "id": "half-scale-q8-1",
        "label": "half-scale-q8-1",
        "disposition": "do_not_reopen",
        "owner": "none",
        "reason": (
            "never screened; OPT-075 forbade a half-scale grid; no measured upside"
        ),
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "pins/opt075_q4_production_admission_contract.json",
        "opt072_ids": [
            "OPT-046:integer_q8_1_w4_half_scale",
            "OPT-062:half_scale_skipped",
        ],
    },
    {
        "id": "OPT-042:cud001_one_warp",
        "label": "historical CUD-001 3e-4 one-warp Q4 diagnostic",
        "disposition": "do_not_reopen",
        "owner": "OPT-085",
        "reason": "do not rebuild; superseded by OPT-085 kernel parity",
        "evidence_path": "fixtures/opt072_rejection_review.json",
        "historical_evidence": "evidence/optimization/opt042-mmv-integer-study/REPORT.md",
        "opt072_ids": ["OPT-042:integer_dp4a_q8block"],
    },
)

CANDIDATE_COUNT = len(REVIEWED_CANDIDATES)


def covered_opt072_ids() -> set[str]:
    ids: set[str] = set()
    for row in REVIEWED_CANDIDATES:
        ids.update(str(item) for item in row.get("opt072_ids") or [])
    return ids


def opt072_by_id(register: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): dict(row) for row in register.get("entries") or []}


def measured_upside(entry: Mapping[str, Any]) -> bool:
    comparison = entry.get("v2_comparison") or {}
    costs = comparison.get("current_complete_cost_ms") or {}
    if not isinstance(costs, Mapping):
        return False
    control = costs.get("packed")
    if not isinstance(control, (int, float)):
        return False
    for key, value in costs.items():
        if key in {"packed", "source"}:
            continue
        if isinstance(value, (int, float)) and float(value) < float(control):
            return True
    return False


def already_implemented(entry: Mapping[str, Any]) -> bool:
    path = str(entry.get("latest_equivalent_path") or "")
    if path in {"", "n/a", "not reinstated", "not launched", "unchanged blocked gate"}:
        return False
    if entry.get("covered_by_opt075_079") is False and not measured_upside(entry):
        return False
    return True


def leftover_qualifies(entry: Mapping[str, Any], *, covered: set[str]) -> bool:
    """True only for an already-implemented numerical-policy leftover with upside."""
    if str(entry.get("id")) in covered:
        return False
    if str(entry.get("kind")) in HARD_ERROR_KINDS:
        return False
    if str(entry.get("disposition")) == "no_repeat":
        return False
    if str(entry.get("next_owner")) in OWNED_ELSEWHERE_OWNERS:
        return False
    numerical = (
        str(entry.get("disposition")) == "numerically_eligible"
        or str(entry.get("kind")) in ARITHMETIC_KINDS
    )
    if not numerical:
        return False
    if str(entry.get("disposition")) not in {"numerically_eligible", "blocked"}:
        return False
    if not already_implemented(entry):
        return False
    if not measured_upside(entry):
        return False
    return True


def scan_opt072_leftovers(register: Mapping[str, Any]) -> list[dict[str, Any]]:
    covered = covered_opt072_ids()
    rows: list[dict[str, Any]] = []
    for entry in register.get("entries") or []:
        if leftover_qualifies(entry, covered=covered):
            rows.append(
                {
                    "id": entry["id"],
                    "task": entry.get("task"),
                    "disposition": "reopen",
                    "reason": "numerical-policy leftover with measured upside",
                    "evidence_path": "fixtures/opt072_rejection_review.json",
                    "opt072_ids": [entry["id"]],
                    "measured_upside": True,
                }
            )
    return rows


def require_file(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise AdmissionError(f"missing evidence {relative}")
    return path


def confirm_owned_q4() -> dict[str, Any]:
    opt075 = load_json(require_file("fixtures/opt075_q4_production_admission.json"))
    opt076 = load_json(require_file("fixtures/opt076_q4_reduction.json"))
    opt085 = load_json(require_file("fixtures/opt085_q4_reevaluation.json"))
    if opt075.get("task") != "OPT-075" or opt076.get("task") != "OPT-076":
        raise AdmissionError("OPT-075/076 fixtures drifted")
    if opt085.get("task") != "OPT-085":
        raise AdmissionError("OPT-085 fixture missing")
    return {
        "opt075_production_kept": opt075.get("production_kept"),
        "opt076_production_kept": opt076.get("production_kept"),
        "opt085_selected_path": opt085.get("selected_path"),
        "opt085_shipping_q4_decode": opt085.get("shipping_q4_decode"),
        "confirmed": True,
    }


def confirm_owned_q8_mmq() -> dict[str, Any]:
    opt086 = load_json(require_file("fixtures/opt086_q8_mmq_reevaluation.json"))
    if opt086.get("task") != "OPT-086":
        raise AdmissionError("OPT-086 fixture missing")
    return {
        "q8_selected": opt086.get("q8_selected_path")
        or opt086.get("shipping_q8_layout"),
        "mmq_selected": opt086.get("mmq_selected_path") or opt086.get("shipping_mmq"),
        "confirmed": True,
    }


def confirm_opt077() -> dict[str, Any]:
    fixture = load_json(require_file("fixtures/opt077_gdn_decode.json"))
    verdict = fixture.get("verdict") or {}
    reasons = [str(item) for item in verdict.get("reasons") or []]
    if "component_interval_not_positive" not in reasons:
        raise AdmissionError("OPT-077 missing component_interval_not_positive")
    if verdict.get("production_kept") is True:
        raise AdmissionError("OPT-077 must not be treated as a keep")
    return {
        "verdict": verdict.get("verdict"),
        "reasons": reasons,
        "production_kept": verdict.get("production_kept"),
        "confirmed": True,
    }


def confirm_opt078() -> dict[str, Any]:
    fixture = load_json(require_file("fixtures/opt078_decode_attention.json"))
    verdict = fixture.get("verdict") or {}
    reasons = [str(item) for item in verdict.get("reasons") or []]
    if verdict.get("verdict") != "performance_rejected":
        raise AdmissionError("OPT-078 verdict is not performance_rejected")
    if "negative_complete_saving" not in reasons:
        raise AdmissionError("OPT-078 missing negative_complete_saving")
    return {
        "verdict": verdict.get("verdict"),
        "reasons": reasons,
        "production_kept": verdict.get("production_kept"),
        "confirmed": True,
    }


def confirm_opt079() -> dict[str, Any]:
    fixture = load_json(require_file("fixtures/opt079_attention_kv_operands.json"))
    verdict = fixture.get("verdict") or {}
    if verdict.get("verdict") != "keep":
        raise AdmissionError("OPT-079 kv_once is not a keep")
    if fixture.get("shipping_attention_pipeline") != "kv_once":
        raise AdmissionError("OPT-079 shipping path is not kv_once")
    return {
        "verdict": verdict.get("verdict"),
        "shipping": fixture.get("shipping_attention_pipeline"),
        "production_kept": verdict.get("production_kept"),
        "confirmed": True,
    }


def confirm_no_repeat(register: Mapping[str, Any], task: str) -> dict[str, Any]:
    entries = [
        row
        for row in register.get("entries") or []
        if row.get("task") == task and row.get("disposition") == "no_repeat"
    ]
    if not entries:
        raise AdmissionError(f"{task} is not an OPT-072 no_repeat loser")
    historical: list[str] = []
    for row in entries:
        for rel in row.get("historical_files") or []:
            require_file(str(rel))
            historical.append(str(rel))
    return {
        "disposition": "no_repeat",
        "historical_files": historical,
        "reevaluation": (entries[0].get("reevaluation") or {}).get("status"),
        "confirmed": True,
    }


def confirm_half_scale(register: Mapping[str, Any]) -> dict[str, Any]:
    legacy = register.get("independent_legacy_candidate") or {}
    if legacy.get("case_id") != "half-scale-q8-1":
        raise AdmissionError("OPT-072 independent legacy candidate is not half-scale")
    if legacy.get("screen_executed") is True:
        raise AdmissionError("half-scale-q8-1 was screened; inventory is stale")
    opt075 = load_json(
        require_file("pins/opt075_q4_production_admission_contract.json")
    )
    opt085 = load_json(require_file("pins/opt085_q4_reevaluation_contract.json"))
    if opt075.get("half_scale_forbidden") is not True:
        raise AdmissionError("OPT-075 no longer forbids a half-scale grid")
    if opt085.get("half_scale_forbidden") is not True:
        raise AdmissionError("OPT-085 no longer forbids a half-scale grid")
    return {
        "screen_executed": False,
        "half_scale_forbidden": True,
        "measured_upside": False,
        "confirmed": True,
    }


def confirm_cud001(register: Mapping[str, Any]) -> dict[str, Any]:
    by_id = opt072_by_id(register)
    entry = by_id.get("OPT-042:integer_dp4a_q8block")
    if not entry:
        raise AdmissionError("OPT-042 CUD-001 inventory row missing")
    if entry.get("disposition") != "already_superseded":
        raise AdmissionError("OPT-042 is no longer already_superseded")
    require_file("evidence/optimization/opt042-mmv-integer-study/REPORT.md")
    return {
        "disposition": entry.get("disposition"),
        "next_owner": entry.get("next_owner"),
        "rebuild": False,
        "confirmed": True,
    }


def historical_opt072_intact() -> dict[str, Any]:
    if not OPT072_FIXTURE.is_file() or not OPT072_REPORT.is_file():
        raise AdmissionError("historical OPT-072 artifacts missing")
    fixture = load_json(OPT072_FIXTURE)
    text = OPT072_REPORT.read_text(encoding="utf-8")
    unmodified = (
        fixture.get("task") == "OPT-072"
        and fixture.get("status") == "rejection_review"
        and "OPT-072" in text
    )
    return {
        "opt072_report": str(OPT072_REPORT.relative_to(ROOT)),
        "opt072_fixture": str(OPT072_FIXTURE.relative_to(ROOT)),
        "opt072_task": fixture.get("task"),
        "opt072_status": fixture.get("status"),
        "entry_count": fixture.get("entry_count"),
        "unmodified": unmodified,
    }


def confirm_candidate(
    spec: Mapping[str, Any], register: Mapping[str, Any]
) -> dict[str, Any]:
    cid = str(spec["id"])
    require_file(str(spec["evidence_path"]))
    if spec.get("historical_evidence"):
        require_file(str(spec["historical_evidence"]))
    by_id = opt072_by_id(register)
    for opt_id in spec.get("opt072_ids") or []:
        if opt_id not in by_id:
            raise AdmissionError(f"{cid} missing OPT-072 id {opt_id}")
    evidence: dict[str, Any]
    if cid in {"OPT-075:q4_integer", "OPT-076:q4_late"}:
        evidence = confirm_owned_q4()
    elif cid in {"OPT-064:q8_layout", "OPT-066:mmq_x_pipeline"}:
        evidence = confirm_owned_q8_mmq()
    elif cid == "OPT-077:tiled_decode_gdn":
        evidence = confirm_opt077()
    elif cid == "OPT-078:prepared_q_veckv":
        evidence = confirm_opt078()
    elif cid == "OPT-079:kv_once":
        evidence = confirm_opt079()
    elif spec.get("opt072_task") in NO_REPEAT_TASKS:
        evidence = confirm_no_repeat(register, str(spec["opt072_task"]))
    elif cid == "half-scale-q8-1":
        evidence = confirm_half_scale(register)
    elif cid == "OPT-042:cud001_one_warp":
        evidence = confirm_cud001(register)
    else:
        raise AdmissionError(f"no confirmer for {cid}")
    if spec["disposition"] not in DISPOSITIONS:
        raise AdmissionError(f"{cid} invalid disposition {spec['disposition']}")
    if spec["disposition"] == "reopen":
        raise AdmissionError(f"pre-documented {cid} must not silently reopen")
    return {
        "id": cid,
        "label": spec["label"],
        "disposition": spec["disposition"],
        "owner": spec.get("owner", "none"),
        "reason": spec["reason"],
        "evidence_path": spec["evidence_path"],
        "historical_evidence": spec.get("historical_evidence"),
        "opt072_ids": list(spec.get("opt072_ids") or []),
        "status_note": spec.get("status_note"),
        "confirmation": evidence,
        "reopen": False,
    }


def select_reopen(
    confirmed: Sequence[Mapping[str, Any]], leftovers: Sequence[Mapping[str, Any]]
) -> str | None:
    reopen = [row for row in confirmed if row.get("disposition") == "reopen"]
    reopen.extend(leftovers)
    if len(reopen) > 1:
        raise AdmissionError(
            "at most one extra already-implemented candidate may be reopened"
        )
    if not reopen:
        return None
    return str(reopen[0]["id"])


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "cases": int(workload.get("cases", CANDIDATE_COUNT)),
        "candidates": int(workload.get("candidates", 1)),
        "tier": str(workload.get("tier", phase)),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "gpu_work": bool(workload.get("gpu_work", False)),
    }


def planned_observation(plan: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-087",
        "warmups": plan["warmups"],
        "samples": samples,
        "observed_warmups": plan["warmups"],
        "observed_samples": samples,
        "observed_candidates": plan["candidates"],
        "observed_shapes": plan["cases"],
        "observed_tier": plan["tier"],
        "pairs": int(plan.get("control_candidate_pairs", 1)),
        "sample_ids": list(range(samples)),
        "acceptance_executed": str(plan["tier"]) == "acceptance",
        "keep": False,
    }


def evaluate_inventory() -> dict[str, Any]:
    register = load_json(OPT072_FIXTURE)
    if register.get("task") != "OPT-072":
        raise AdmissionError("OPT-072 inventory is not readable")
    confirmed = [confirm_candidate(spec, register) for spec in REVIEWED_CANDIDATES]
    leftovers = scan_opt072_leftovers(register)
    reopened = select_reopen(confirmed, leftovers)
    if reopened is not None:
        raise AdmissionError(
            f"unexpected reopen candidate {reopened}; stop rather than expand"
        )
    intact = historical_opt072_intact()
    if not intact["unmodified"]:
        raise AdmissionError("historical OPT-072 report/fixture drifted")
    if CANDIDATE_COUNT != len(confirmed):
        raise AdmissionError("candidate table length drifted")
    if OPT074_FAMILY_ADMISSION_REQUIRED:
        raise AdmissionError("opt074_family_admission_required must stay false")
    return {
        "candidates": confirmed,
        "leftover_qualifying": leftovers,
        "reopened_candidate": None,
        "no_additional_reopen": True,
        "historical_opt072": intact,
        "opt074_coverage_unadmitted_blocker": False,
        "opt074_family_admission_required": False,
        "production_pins_unchanged": True,
        "claims_throughput": False,
        "gpu_work": False,
    }


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in payload.get("candidates") or []:
        lines.append(
            f"| {row.get('id')} | {row.get('disposition')} | "
            f"{row.get('reason')} | `{row.get('evidence_path')}` |"
        )
    leftover = payload.get("leftover_qualifying") or []
    leftover_text = "none" if not leftover else json.dumps(leftover)
    text = f"""# OPT-087 — Historical numerical-policy leftover reevaluation

Status: **no_additional_reopen**. Authority llama.cpp `{LLAMA_REV}`, GGUF
SHA-256 `{GGUF_SHA}`. Inventory of remaining OPT-072 leftovers plus
OPT-075–079 outcomes. At most one extra already-implemented candidate may be
reopened; none qualified. Sequence remains kernel parity → model quality →
performance. No large sweep. No production pin change.

`opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required={OPT074_FAMILY_ADMISSION_REQUIRED}`.
`claims_throughput={payload.get("claims_throughput", False)}`.

## Disposition table

| Candidate | Disposition | Reason | Evidence |
|---|---|---|---|
{chr(10).join(lines)}

`no_additional_reopen={payload.get("no_additional_reopen", True)}`.
`reopened_candidate={payload.get("reopened_candidate")}`.
Qualifying OPT-072 leftovers not already tabled: {leftover_text}.

## Confirmed non-reopens

- OPT-075/076 Q4 integer and late-reduction remain owned by OPT-085.
- OPT-064/066 Q8/MMQ remain owned by OPT-086.
- OPT-077 tiled decode GDN stays sequential (`component_interval_not_positive`).
- OPT-078 prepared_q / veckv stays `performance_rejected` with negative
  complete saving. Do not rerun hoping the interval flips.
- OPT-079 `kv_once` remains the keep; OPT-088 checks interactions.
- OPT-067/065/068/024/027–030/037/054/055 stay OPT-072 `no_repeat`
  performance losers. Missing OPT-074 coverage does not make them eligible.
- `half-scale-q8-1` was never screened; OPT-075/085 forbid a half-scale grid.
- Historical CUD-001 3e-4 one-warp Q4 diagnostic is not rebuilt.

Historical OPT-072 report
`{payload.get("historical_opt072", {}).get("opt072_report")}` is unmodified.

## tok/s

Official tok/s delta versus the then-current shipping baseline is **0**.
`claims_throughput` is true only when a reopened candidate is kept.
"""
    REPORT.write_text(text, encoding="utf-8")


def build_fixture(
    evaluated: Mapping[str, Any], *, mode: str, phase: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-087",
        "mode": mode,
        "phase": phase,
        "status": "no_additional_reopen",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "token_generator": TOKEN_GENERATOR,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "no_additional_reopen": True,
        "reopened_candidate": None,
        "at_most_one_reopen": True,
        "candidate_count": len(evaluated["candidates"]),
        "candidates": list(evaluated["candidates"]),
        "leftover_qualifying": list(evaluated["leftover_qualifying"]),
        "dispositions": list(DISPOSITIONS),
        "opt074_family_admission_required": False,
        "opt074_coverage_unadmitted_blocker": False,
        "historical_opt072_unmodified": True,
        "historical_opt072": evaluated["historical_opt072"],
        "production_pins_unchanged": True,
        "gpu_work": False,
        "host_only": True,
        "measurement_utc": utc_now(),
        "report_path": "evidence/optimization/opt087-historical-reevaluation/REPORT.md",
    }


def run(
    mode: str,
    phase: str,
    run_dir: Path,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    plan = family_plan(phase, mode)
    if int(plan["cases"]) != CANDIDATE_COUNT:
        raise AdmissionError(
            f"{phase} cases {plan['cases']} != reviewed {CANDIDATE_COUNT}"
        )
    if plan.get("gpu_work"):
        raise AdmissionError("inventory/review must stay host_only without a reopen")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_inventory()
    fixture = build_fixture(evaluated, mode=mode, phase=phase)
    write_report(fixture)
    dump_json(FIXTURE, fixture)
    observed = planned_observation(plan)
    payload = {
        "schema_version": 1,
        "task": "OPT-087",
        "mode": mode,
        "phase": phase,
        "success": True,
        "result_class": "ok",
        "gpu_work": False,
        "host_only": True,
        "no_additional_reopen": True,
        "reopened_candidate": None,
        "candidate_count": CANDIDATE_COUNT,
        "claims_throughput": False,
        "opt074_coverage_unadmitted_blocker": False,
        "historical_opt072_unmodified": True,
        "production_pins_unchanged": True,
        "native_counts": observed,
        "keep": False,
        "fixture": "fixtures/opt087_historical_reevaluation.json",
        "report": "evidence/optimization/opt087-historical-reevaluation/REPORT.md",
        **observed,
    }
    dump_json(run_dir / "opt087-result.json", payload)
    dump_json(run_dir / "opt087_historical_reevaluation.json", fixture)
    print("QW38_OPT087_RESULT=" + json.dumps(payload))
    return fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="inventory")
    parser.add_argument(
        "--mode", choices=("feedback", "acceptance"), default="feedback"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-087" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
