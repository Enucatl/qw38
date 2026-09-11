"""Host helpers for OPT-072 correctness-rejection inventory and Q8_1 audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.production_numerics import STRICT_CUD001_ABS, STRICT_CUD001_RMS  # noqa: E402
from tools.production_numerics_v2 import (  # noqa: E402
    INTEGER_SUM_CASES,
    STAGING_LLAMA_SUM_X,
    STAGING_QUARTZ_SUM_Q,
    V2_ABS_HEADROOM,
    half_round,
    integer_lost_in_half,
    v2_abs_rms_ceiling,
)
from tools.run_optimization_task import loop_product  # noqa: E402

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CONTRACT = ROOT / "pins/opt072_rejection_review_contract.json"
ITERATION = ROOT / "pins/opt072_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt072_rejection_review.json"
REPORT = ROOT / "evidence/optimization/opt072-rejection-review/REPORT.md"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
DOTS = ROOT / "cuda/q4k_decode_dots.cuh"
NATIVE_OPT059 = "build/qw38-cuda-opt059-numerics-test"
NATIVE_OPT062 = "build/qw38-cuda-opt062-q4-admission-test"
NATIVE_OPT063 = "build/qw38-cuda-opt063-integer-ffn-test"

DISPOSITIONS = (
    "still_valid",
    "evidence_invalid",
    "numerically_eligible",
    "already_superseded",
    "blocked",
    "kept",
    "no_repeat",
)
CLASSIFICATIONS = (
    "already_superseded",
    "wrong_reference",
    "incomplete_evidence",
    "real_correctness_bug",
    "performance_only",
    "mixed",
    "kept",
    "instrumentation",
    "blocked_gate",
    "superseded_task",
)
KINDS = (
    "numeric_rejection",
    "missing_admission",
    "exact_scalar_association",
    "invalid_prompt_scorer",
    "missing_authority",
    "stale_object_selector",
    "layout_nonfinite_state",
    "performance_only",
    "kept",
    "instrumentation",
    "blocked_gate",
    "superseded_task",
)
HARD_ERROR_KINDS = frozenset({"layout_nonfinite_state", "stale_object_selector"})
TASK_BEGIN = 1
TASK_END = 69
OPT046_CUD001_ABS = 3.05175781e-4
OPT059_SYNTHETIC_LLAMA_ABS = 19.0954106
OPT062_PACKED_MS = 15.7643099
OPT062_INTEGER_MS = 12.4760742
OPT063_PAIRED_MS = 11.2725649

INSTRUMENTATION_TASKS = {
    "OPT-001": "synchronized timings and NVTX attribution",
    "OPT-014": "2K prefill attribution",
    "OPT-015": "2K recovery ranking",
    "OPT-020": "mixer versus core split attribution",
    "OPT-021": "4K keep/reject oracle",
    "OPT-032": "decode oracle and sink attribution",
    "OPT-038": "post-ladder gap map",
    "OPT-043": "component gap versus llama",
    "OPT-044": "v1 production numerics policy freeze",
    "OPT-057": "bounded optimization runner",
    "OPT-060": "engine family attribution",
    "OPT-061": "component replay",
}

KEPT_TASKS = {
    "OPT-002": ("justified fusions", "fused production paths"),
    "OPT-003": ("stable-address CUDA graphs", "graph/non-graph equality"),
    "OPT-004": ("row-bucket dispatch table", "selected_mmv_warps"),
    "OPT-005": ("tiled causal prompt attention", "fattn production"),
    "OPT-006": ("GQA shared KV loads", "gqa attention"),
    "OPT-007": ("multi-row prompt attention blocks", "query-row tiles"),
    "OPT-008": ("4096-token prompt chunk", "kPromptChunkRows"),
    "OPT-009": ("batched Q8/Q4/Q6 prompt MMQ", "launch_quant_mmq"),
    "OPT-010": ("tiled KV layout", "kv tile storage"),
    "OPT-011": ("prompt pipeline/fusion", "prompt overlap"),
    "OPT-012": ("prompt CUDA graphs", "4096 FFN graphs"),
    "OPT-013": ("associative GDN prompt scan", "parallel GDN scan"),
    "OPT-017": ("mixer Q8_0 MMA MMQ", "launch_q8_mmq_bf16 MMA"),
    "OPT-019": ("GDN/attention core recovery", "warp-column GDN / fattn"),
    "OPT-023": ("skinny mixer dispatch", "alpha/beta MMV"),
    "OPT-025": ("FFN shared-Y", "shared Q8_1 Y"),
    "OPT-026": ("fattn stream-K occupancy", "grid.z=2 bipartition"),
    "OPT-033": ("register-resident VKQ", "fattn register sums"),
    "OPT-034": ("packed Q4/Q6 MMV loads", "kSelectedMmvLoadPath=packed"),
    "OPT-035": ("probability times V MMA", "fattn PV MMA"),
    "OPT-036": ("decode KV partitions", "16 partitions"),
    "OPT-039": ("warp-owned decode attention", "warp query heads"),
    "OPT-040": ("hoisted GDN inverses", "shared inverse norms"),
    "OPT-041": ("warp-owned prompt QK", "microtile ownership"),
    "OPT-045": ("parallel RMSNorm", "cooperative fmaf"),
    "OPT-047": ("Q8_0 mixer decode DP4A", "dp4a_q8_1"),
    "OPT-048": ("Q6_K full-vocab integer dots", "integer_q8_1"),
    "OPT-049": ("paired_staged decode FFN", "paired_staged"),
    "OPT-050": ("hoisted query prepare", "prepare once"),
    "OPT-051": ("prompt attention pipeline", "F16 async pipeline"),
    "OPT-052": ("GDN arithmetic hoist", "scaled Q/K and decay"),
    "OPT-053": ("MMQ FMA/async Y pipeline", "fma_async"),
}

PERFORMANCE_NO_REPEAT = {
    "OPT-024": (
        "Blackwell Q8 D2R",
        "4K oracle did not beat OPT-023; A/B lost to quality MMA",
        "evidence/optimization/opt024-mixer-q8-d2r/REJECTION.md",
        "OPT-079",
    ),
    "OPT-027": (
        "persistent Ada+ fattn stream-K",
        "4K oracle did not beat OPT-026",
        "evidence/optimization/opt027-persistent-fattn/REJECTION.md",
        "none",
    ),
    "OPT-028": (
        "Q4/Q6 MMQ stream-K",
        "4K oracle did not beat OPT-026",
        "evidence/optimization/opt028-mmq-streamk/REJECTION.md",
        "none",
    ),
    "OPT-029": (
        "fused GDN conv/gated output",
        "4K oracle did not beat OPT-026; decode GDN remains sequential",
        "evidence/optimization/opt029-gdn-fuse/REJECTION.md",
        "OPT-077",
    ),
    "OPT-030": (
        "PDL prompt launches",
        "4K oracle did not beat OPT-026",
        "evidence/optimization/opt030-pdl-launches/REJECTION.md",
        "none",
    ),
    "OPT-037": (
        "FFN I/J tile sweep",
        "every projection retained i128_j128; keep sitting skipped",
        "evidence/optimization/opt037-ffn-tiles/REJECTION.md",
        "none",
    ),
    "OPT-054": (
        "prefill microbatch 512/1024/2048",
        "complete P no-change; atomic 4096 retained",
        "evidence/optimization/opt054-prefill-microbatch/REPORT.md",
        "none",
    ),
    "OPT-055": (
        "broader execution graphs",
        "measured idle about 0.1 ms/token; FFN-only graphs retained",
        "evidence/optimization/opt055-execution-graphs/REPORT.md",
        "OPT-071",
    ),
    "OPT-065": (
        "MMQ I/J retune with FMA/async Y",
        "complete FFN lost versus i128_j128",
        "evidence/optimization/opt065-mmq-tiles/REPORT.md",
        "none",
    ),
    "OPT-067": (
        "prompt gate/up pairing",
        "complete FFN 11.28 ms vs control 8.52 ms",
        "evidence/optimization/opt067-prompt-pair/REPORT.md",
        "none",
    ),
    "OPT-068": (
        "scoped O3/FMA codegen",
        "primitive numerics passed; complete FFN slower than O2 --fmad=false",
        "evidence/optimization/opt068-scoped-codegen/REPORT.md",
        "none",
    ),
}


def rint_to_int(value: float) -> int:
    """Match CUDA rintf for the integer-sum cases used here."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def quants_for_sum(total: int, lanes: int = 32) -> list[int]:
    sign = 1 if total >= 0 else -1
    remaining = abs(total)
    values = [0] * lanes
    for index in range(lanes):
        take = min(127, remaining)
        values[index] = sign * take
        remaining -= take
    if remaining != 0:
        raise ValueError(f"cannot encode integer sum {total} in {lanes} int8 lanes")
    return values


def exact_int32_sum_q(quants: Sequence[int]) -> int:
    return int(sum(int(value) for value in quants))


def half_sum_q(quants: Sequence[int]) -> float:
    return half_round(float(exact_int32_sum_q(quants)))


def half_sum_x(values: Sequence[float]) -> float:
    return half_round(float(sum(values)))


def q4_sum_q_consumer(stored_half: float) -> int:
    return rint_to_int(float(stored_half))


def pairing_legal(producer: str, consumer: str) -> bool:
    if producer == STAGING_LLAMA_SUM_X and consumer == STAGING_QUARTZ_SUM_Q:
        return False
    return True


def integer_sum_audit(total: int, *, scale: float = 1.0) -> dict[str, Any]:
    quants = quants_for_sum(total)
    activations = [float(quant) * float(scale) for quant in quants]
    exact_q = exact_int32_sum_q(quants)
    stored_q = half_sum_q(quants)
    stored_x = half_sum_x(activations)
    consumed_q = q4_sum_q_consumer(stored_q)
    consumed_x = q4_sum_q_consumer(stored_x)
    lost = integer_lost_in_half(total)
    return {
        "integer": total,
        "exact_int32_sum_q": exact_q,
        "half_sum_q": stored_q,
        "half_sum_x": stored_x,
        "q4_consumer_from_sum_q": consumed_q,
        "q4_consumer_from_sum_x": consumed_x,
        "units_lost_in_half": lost,
        "sum_q_consumer_matches_exact": consumed_q == exact_q,
        "sum_x_producer_feeds_sum_q_consumer": not pairing_legal(
            STAGING_LLAMA_SUM_X, STAGING_QUARTZ_SUM_Q
        ),
        "scale": scale,
    }


def may_admit_by_numerical_headroom(entry: Mapping[str, Any]) -> bool:
    kind = str(entry.get("kind", ""))
    classification = str(entry.get("classification", ""))
    if kind in HARD_ERROR_KINDS:
        return False
    if classification == "real_correctness_bug":
        return False
    if entry.get("nonfinite", 0):
        return False
    if entry.get("layout_error"):
        return False
    return True


def _base(
    task: str,
    variant: str,
    *,
    kind: str,
    classification: str,
    disposition: str,
    original_test: str,
    reference: str,
    metric: str,
    envelope: str,
    source: str,
    latest: str,
    owner: str,
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": f"{task}:{variant}",
        "task": task,
        "variant": variant,
        "kind": kind,
        "classification": classification,
        "disposition": disposition,
        "original_test": original_test,
        "original_reference": reference,
        "metric": metric,
        "envelope": envelope,
        "source_candidate": source,
        "latest_equivalent_path": latest,
        "current_disposition": disposition,
        "next_owner": owner,
        "claims_throughput": False,
        "admit_by_numerical_headroom": False,
        "historical_30_sample_oracle": False,
        "candidate_fitted_epsilon": False,
    }
    entry.update(extra)
    if "reevaluation" not in entry:
        if disposition == "kept":
            entry["reevaluation"] = {"status": "superseding_keep"}
        elif disposition == "no_repeat":
            entry["reevaluation"] = {
                "status": "performance_loss_not_rescued_by_tolerance"
            }
        elif disposition == "blocked":
            entry["reevaluation"] = {
                "status": "unresolved_dependency",
                "follow_up": owner,
            }
        elif disposition == "already_superseded":
            entry["reevaluation"] = {"status": "proven_superseding_equivalence"}
        elif disposition == "still_valid":
            entry["reevaluation"] = {"status": "reproduced_from_retained_evidence"}
        elif disposition == "numerically_eligible":
            entry["reevaluation"] = {
                "status": "unresolved_dependency",
                "follow_up": owner,
            }
        else:
            entry["reevaluation"] = {"status": "not_a_correctness_candidate"}
    entry["admit_by_numerical_headroom"] = may_admit_by_numerical_headroom(entry)
    return entry


def _kept(task: str, variant: str, latest: str, note: str) -> dict[str, Any]:
    return _base(
        task,
        variant,
        kind="kept",
        classification="kept",
        disposition="kept",
        original_test="production keep sitting",
        reference="task-owned reference retained",
        metric="keep/reject oracle of that increment",
        envelope="frozen task envelopes",
        source=variant,
        latest=latest,
        owner="none",
        note=note,
        reevaluation={
            "status": "superseding_keep",
            "gpu_authority": "not_required_for_keep_inventory",
        },
    )


def _instrumentation(task: str, note: str) -> dict[str, Any]:
    return _base(
        task,
        "primary",
        kind="instrumentation",
        classification="instrumentation",
        disposition="kept",
        original_test="diagnostic/protocol task",
        reference="n/a",
        metric="n/a",
        envelope="no numerical admission",
        source="primary",
        latest="current diagnostics",
        owner="none",
        note=note,
        reevaluation={"status": "not_a_correctness_candidate"},
    )


def _performance(
    task: str, variant: str, reason: str, evidence: str, owner: str
) -> dict[str, Any]:
    return _base(
        task,
        variant,
        kind="performance_only",
        classification="performance_only",
        disposition="no_repeat",
        original_test="paired A/B or complete-cost screen",
        reference="then-current keep/control",
        metric="complete component or 4K oracle tok/s",
        envelope="performance gate; numerics already admitted or unused",
        source=variant,
        latest="not reinstated",
        owner=owner,
        note=reason,
        historical_files=[evidence],
        no_repeat_reason=reason,
        reevaluation={
            "status": "performance_loss_not_rescued_by_tolerance",
            "gpu_authority": "not_applicable",
        },
    )


def _v2_unadmitted() -> dict[str, Any]:
    budget = v2_abs_rms_ceiling(STRICT_CUD001_ABS, None)
    return {
        "original_rule": "CUD-001 max_abs 3e-4 RMS 2e-4 vs scalar host Q8-staged MMV",
        "v2_rule": "max(strict, 1.25 * measured_llama_gpu_error + 1e-6)",
        "v2_headroom": V2_ABS_HEADROOM,
        "llama_gpu_error_production_k": None,
        "v2_ceiling_production_k": budget["ceiling"],
        "v2_admitted_production_k": False,
        "copied_synthetic_m17_k256_to_production_k": False,
        "relaxation_changes_eligibility": False,
        "reason": (
            "production-K llama GPU error is unadmitted; synthetic M17/K256 "
            f"llama GPU vs FP64 original max_abs {OPT059_SYNTHETIC_LLAMA_ABS} "
            "is Q4 quantization error, not integer-vs-packed association, and "
            "must not be copied onto production K"
        ),
    }


def build_register() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for task, note in INSTRUMENTATION_TASKS.items():
        entries.append(_instrumentation(task, note))
    for task, (variant, latest) in KEPT_TASKS.items():
        entries.append(_kept(task, variant, latest, "installed keep"))
    for task, (variant, reason, evidence, owner) in PERFORMANCE_NO_REPEAT.items():
        entries.append(_performance(task, variant, reason, evidence, owner))

    entries.append(
        _base(
            "OPT-016",
            "2k_parity_gate",
            kind="blocked_gate",
            classification="blocked_gate",
            disposition="blocked",
            original_test="cold exact-2048 tok/s versus llama-bench",
            reference="pinned llama.cpp cc83d7b",
            metric="tok/s",
            envelope="Quartz >= llama 2K",
            source="production",
            latest="unchanged blocked gate",
            owner="OPT-080",
            note="performance parity gate; not a numerical rejection",
        )
    )
    entries.append(
        _base(
            "OPT-018",
            "q4k_q6k_mma_mmq",
            kind="kept",
            classification="kept",
            disposition="kept",
            original_test="ds4 Q4 association vs CPU dequant GEMM",
            reference="host dequant-weight x BF16 activation",
            metric="abs>0.20*sqrt(K) and rel>0.05",
            envelope="exact_scalar_association historical; CUD-002 variant retained",
            source="quality MMA MMQ",
            latest="launch_quant_mmq_mma",
            owner="none",
            kind_detail="exact_scalar_association",
            note=(
                "historical association work stays the prompt-MMQ keep; v2 GPU "
                "budgets do not rewrite this installed path"
            ),
        )
    )
    entries.append(
        _base(
            "OPT-018",
            "association_rule_historical",
            kind="exact_scalar_association",
            classification="wrong_reference",
            disposition="still_valid",
            original_test="element fails only if abs>0.20*sqrt(K) and rel>0.05",
            reference="CPU dequant GEMM",
            metric="paired abs/rel",
            envelope="plan.md option C association; not OPT-059 v2",
            source="historical MMQ admission",
            latest="unchanged installed MMA",
            owner="none",
            note="association remains a historical diagnostic, not a v2 budget",
            reevaluation={"status": "historical_rule_retained_on_installed_keep"},
        )
    )
    entries.append(
        _base(
            "OPT-022",
            "q8_association_quality_mmq",
            kind="kept",
            classification="kept",
            disposition="kept",
            original_test="plan.md Q8 association",
            reference="OPT-009 byte-exact Q8 reference",
            metric="association plus 4K oracle",
            envelope="Q8 association historical",
            source="quality_mma I=128 J=128",
            latest="mixer quality MMA",
            owner="none",
        )
    )
    entries.append(
        _base(
            "OPT-031",
            "mixer_gdn_4096_graphs",
            kind="superseded_task",
            classification="superseded_task",
            disposition="already_superseded",
            original_test="never implemented as a keep",
            reference="n/a",
            metric="n/a",
            envelope="n/a",
            source="ledger row",
            latest="OPT-055 FFN-only graphs",
            owner="none",
            note="superseded by OPT-055; no speedup claimed",
            reevaluation={"status": "proven_superseding_equivalence"},
        )
    )
    entries.append(
        _base(
            "OPT-042",
            "integer_dp4a_q8block",
            kind="numeric_rejection",
            classification="wrong_reference",
            disposition="already_superseded",
            original_test="tests/test_opt042_mmv_integer_study.py CUD-001",
            reference="packed FP32 Q8Block host/scalar envelope",
            metric="max_abs/RMS vs packed/scalar",
            envelope="CUD-001 3e-4 / 2e-4",
            source="cuda/opt042_mmv_integer_study.cu",
            latest="OPT-062 integer_q8 Q8Block cooperative dots",
            owner="OPT-075",
            historical_files=[
                "fixtures/opt042_mmv_integer_study.json",
                "evidence/optimization/opt042-mmv-integer-study/REPORT.md",
            ],
            native_targets=[NATIVE_OPT059, NATIVE_OPT062],
            v2_comparison=_v2_unadmitted(),
            original_failure={
                "shape": "q4_k_17x256",
                "rows": 17,
                "columns": 256,
                "admissibility": "numeric_reject",
                "staging_equal": True,
            },
            reevaluation={
                "status": "proven_superseding_equivalence",
                "strict": "historical CUD-001 reject retained",
                "fp64_original": "not reproduced with 30-sample oracle",
                "fp64_staged": "current OPT-062 integer vs staged abs 0",
                "gpu_authority": "unadmitted_route_OPT-074",
            },
            note="do not rebuild the one-warp diagnostic; review via OPT-062/075",
        )
    )
    entries.append(
        _base(
            "OPT-046",
            "integer_q8_w4",
            kind="numeric_rejection",
            classification="mixed",
            disposition="numerically_eligible",
            original_test="tests/test_cuda_quant_mmv.py q4_k_17x256 CUD-001",
            reference="scalar/host Q8-staged packed MMV",
            metric="max_abs",
            envelope="CUD-001 3e-4",
            source="integer_q8_w4 cooperative DP4A",
            latest="OPT-062 integer_q8 all three FFN legs",
            owner="OPT-075",
            historical_files=[
                "fixtures/opt046_q4_decode.json",
                "evidence/optimization/opt046-q4-decode/REJECTION.md",
                "evidence/optimization/opt046-q4-decode/REPORT.md",
            ],
            native_targets=[NATIVE_OPT059, NATIVE_OPT062],
            original_failure={
                "shape": "q4_k_17x256",
                "max_abs": OPT046_CUD001_ABS,
                "envelope": STRICT_CUD001_ABS,
                "rms_envelope": STRICT_CUD001_RMS,
                "packed_max_abs": 0.000244140625,
            },
            v2_comparison={
                **_v2_unadmitted(),
                "faster_implementation_if_admitted": "integer_q8 / integer_q8_w4",
                "current_complete_cost_ms": {
                    "packed": OPT062_PACKED_MS,
                    "integer_q8": OPT062_INTEGER_MS,
                    "source": "OPT-062 rotating complete FFN screen",
                },
            },
            reevaluation={
                "status": "unresolved_dependency",
                "strict": f"historical max_abs {OPT046_CUD001_ABS} fails 3e-4",
                "fp64_original": "OPT-062 vs original bounded after host Q8 reconstruction",
                "fp64_staged": "OPT-062 integer vs staged abs 0",
                "gpu_authority": "unadmitted_route_OPT-074",
                "follow_up": "OPT-075 production admission after OPT-074",
            },
            note=(
                "old CUD-001 reduction-order miss is not the production gate; "
                "missing production-K GPU authority blocks a completed numerical keep"
            ),
        )
    )
    entries.append(
        _base(
            "OPT-046",
            "historical_scheduler_nan",
            kind="layout_nonfinite_state",
            classification="real_correctness_bug",
            disposition="still_valid",
            original_test="tests/test_cuda_full_scheduler.py with integer pin",
            reference="scalar greedy token",
            metric="nonfinite logits",
            envelope="zero nonfinites; greedy match",
            source="integer production pin sitting",
            latest="OPT-058 finite packed scheduler",
            owner="OPT-075",
            nonfinite=248320,
            layout_error=False,
            historical_files=["evidence/optimization/opt046-q4-decode/REJECTION.md"],
            reevaluation={
                "status": "reproduced_from_retained_evidence",
                "note": (
                    "OPT-058 showed current packed eager/graph/trace finite; "
                    "historical integer-pin NaNs remain a hard error if they return"
                ),
            },
        )
    )
    entries.append(
        _base(
            "OPT-046",
            "integer_q8_1_w4_half_scale",
            kind="missing_admission",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="OPT-046 A/B integer_q8_1_w4",
            reference="llama-style Q8_1 half-scale staging",
            metric="complete MMV ms plus CUD-001",
            envelope="OPT-059 format id; v2 unadmitted",
            source="integer_q8_1_w4",
            latest="quantize_bf16_q8_1 + UseQ81 Q4 consumer",
            owner="OPT-072",
            case_id="half-scale-q8-1",
            native_targets=[NATIVE_OPT059, NATIVE_OPT062],
            covered_by_opt075_079=False,
            note=(
                "half-scale skipped by OPT-062 because OPT-059 v2_admitted=false; "
                "OPT-075 forbids a half-scale grid. Enumerated here as the independent "
                "legacy numerical candidate with one control/candidate screen."
            ),
            reevaluation={
                "status": "unresolved_dependency",
                "gpu_authority": "unadmitted_route_OPT-074",
                "screen_executed": False,
                "qualifying_result": "screened_in_follow_up_only",
            },
            v2_comparison=_v2_unadmitted(),
        )
    )
    entries.append(
        _base(
            "OPT-047",
            "q8_1_sum_field_unused",
            kind="kept",
            classification="kept",
            disposition="kept",
            original_test="Q8_0 x Q8_1 DP4A vs direct BF16",
            reference="q8_mmv_bf16",
            metric="dot uses scale*values only",
            envelope="OPT-044 vs original BF16",
            source="dp4a_q8_1",
            latest="OPT-064 r2_w2",
            owner="OPT-070",
            note="Q8 consumer does not read stored q8_sum; preserve production Q8 staging",
            q8_1_consumer="values_and_scale_only",
        )
    )
    entries.append(
        _base(
            "OPT-048",
            "q6_uses_q8_1_values_not_sum",
            kind="kept",
            classification="kept",
            disposition="kept",
            original_test="full-vocab Q6 integer dots",
            reference="packed FP32 logits",
            metric="dot uses Q8_1 scale and qs",
            envelope="OPT-044 full vocab",
            source="integer_q8_1_w2",
            latest="integer_q8_1",
            owner="none",
            note="Q6 consumer does not read stored q8_sum; preserve production Q6 staging",
            q8_1_consumer="values_and_scale_only",
        )
    )
    entries.append(
        _base(
            "OPT-049",
            "bf16_activation_load_uninstalled",
            kind="numeric_rejection",
            classification="wrong_reference",
            disposition="already_superseded",
            original_test="paired-only BF16 activation vs Q8 staging",
            reference="separate-leg packed Q4",
            metric="max_abs ~0.02-0.03",
            envelope="must match Q8 staged control",
            source="paired BF16 activation loads",
            latest="paired_staged keep",
            owner="none",
            note="variant inside kept task; Q8 staging path was installed instead",
            reevaluation={"status": "proven_superseding_equivalence"},
        )
    )
    entries.append(
        _base(
            "OPT-056",
            "plus5_performance_gate",
            kind="blocked_gate",
            classification="blocked_gate",
            disposition="blocked",
            original_test="same-sitting P/D128/D2048 versus llama +5%",
            reference="pinned llama.cpp",
            metric="tok/s and p95",
            envelope="OPT-056 original gates unchanged",
            source="combined production",
            latest="OPT-069 sitting as latest measurement",
            owner="OPT-080",
        )
    )
    entries.append(
        _base(
            "OPT-058",
            "functional_and_held_out_quality",
            kind="invalid_prompt_scorer",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="quality v2 functional prompts and held-out NLL",
            reference="pinned llama quality oracle",
            metric="task_arithmetic / NLL",
            envelope="PPL ratio <=1.01; recurrence NLL <=0.02",
            source="current scheduler",
            latest="pins/production_quality_v2_inputs.json",
            owner="OPT-073",
            note="absolute versus regression quality is OPT-073; do not retune prompts here",
            reevaluation={"status": "unresolved_dependency", "follow_up": "OPT-073"},
        )
    )
    entries.append(
        _base(
            "OPT-059",
            "production_k_gpu_authority",
            kind="missing_authority",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="llama GPU projection export",
            reference="pinned llama GPU dispatch",
            metric="max_abs/RMS vs FP64 original",
            envelope="v2 reference-derived ceilings",
            source="synthetic Q4 17x256 export only",
            latest="pins/production_numerics_v2_contract.json",
            owner="OPT-074",
            note="missing production-K GPU references stay unadmitted",
            reevaluation={"status": "unresolved_dependency", "follow_up": "OPT-074"},
            v2_comparison=_v2_unadmitted(),
        )
    )
    entries.append(
        _base(
            "OPT-062",
            "integer_q8_complete_ffn",
            kind="missing_admission",
            classification="incomplete_evidence",
            disposition="numerically_eligible",
            original_test="build/qw38-cuda-opt062-q4-admission-test",
            reference="staged FP64 and packed control",
            metric="abs vs staged/packed; complete FFN ms",
            envelope="OPT-059 v2 unadmitted; strict CUD-001 retained separately",
            source="integer_q8 4 warps/row",
            latest="integer_q8 diagnostic dispatch",
            owner="OPT-075",
            historical_files=[
                "fixtures/opt062_q4_admission.json",
                "evidence/optimization/opt062-q4-admission/REPORT.md",
            ],
            native_targets=[NATIVE_OPT062],
            original_failure=None,
            v2_comparison={
                **_v2_unadmitted(),
                "faster_implementation_if_admitted": "integer_q8",
                "current_complete_cost_ms": {
                    "packed": OPT062_PACKED_MS,
                    "integer_q8": OPT062_INTEGER_MS,
                    "source": "OPT-062 rotating complete FFN screen",
                },
            },
            reevaluation={
                "status": "unresolved_dependency",
                "strict": "historical CUD-001 not the production gate",
                "fp64_original": "bounded after host Q8 reconstruction",
                "fp64_staged": 0.0,
                "gpu_authority": "unadmitted_route_OPT-074",
                "follow_up": "OPT-075",
            },
            note="half-scale skipped; Q8Block prequant never reinterprets as Q8_1",
        )
    )
    entries.append(
        _base(
            "OPT-062",
            "half_scale_skipped",
            kind="missing_admission",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="OPT-062 half-scale candidate list",
            reference="OPT-059 format id",
            metric="n/a skipped",
            envelope="v2_admitted=false",
            source="integer_q8_1 half-scale",
            latest="not launched",
            owner="OPT-072",
            case_id="half-scale-q8-1",
            covered_by_opt075_079=False,
            reevaluation={"status": "unresolved_dependency", "screen_executed": False},
        )
    )
    entries.append(
        _base(
            "OPT-063",
            "paired_integer",
            kind="missing_admission",
            classification="incomplete_evidence",
            disposition="numerically_eligible",
            original_test="build/qw38-cuda-opt063-integer-ffn-test",
            reference="unfused integer_q8 and staged FP64",
            metric="abs 0 vs unfused/staged; complete FFN ms",
            envelope="OPT-062 arithmetic; v2 unadmitted",
            source="paired_integer 4 warps",
            latest="q4k_coop_gate_up_swiglu",
            owner="OPT-075",
            historical_files=[
                "fixtures/opt063_integer_ffn.json",
                "evidence/optimization/opt063-integer-ffn/REPORT.md",
            ],
            native_targets=[NATIVE_OPT063],
            v2_comparison={
                **_v2_unadmitted(),
                "faster_implementation_if_admitted": "paired_integer",
                "current_complete_cost_ms": {
                    "packed": 15.8205442,
                    "integer_q8": 12.3049278,
                    "paired_integer": OPT063_PAIRED_MS,
                    "source": "OPT-063 rotating complete FFN screen",
                },
            },
            reevaluation={
                "status": "unresolved_dependency",
                "fp64_staged": 0.0,
                "gpu_authority": "unadmitted_route_OPT-074",
                "follow_up": "OPT-075",
            },
        )
    )
    entries.append(
        _base(
            "OPT-064",
            "r2_w2_keep_evidence",
            kind="kept",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="OPT-064 screen 1+3 layouts",
            reference="r1_w4 control",
            metric="rotating weighted mixer ms",
            envelope="installed keep with incomplete acceptance",
            source="r2_w2",
            latest="kSelectedQ8 layout r2_w2",
            owner="OPT-070",
            note="installed keep; evidence repair belongs to OPT-070, not a numerical reject",
            reevaluation={"status": "unresolved_dependency", "follow_up": "OPT-070"},
        )
    )
    entries.append(
        _performance(
            "OPT-064",
            "r8_w1",
            "rotating mixer slower than r2_w2; no end-to-end for losing layouts",
            "evidence/optimization/opt064-q8-rows/REPORT.md",
            "none",
        )
    )
    entries.append(
        _base(
            "OPT-066",
            "fma_async_x_keep_evidence",
            kind="kept",
            classification="incomplete_evidence",
            disposition="blocked",
            original_test="OPT-066 layer-0 mean screen",
            reference="fma_async",
            metric="complete FFN ms",
            envelope="installed keep with incomplete acceptance",
            source="fma_async_x",
            latest="kSelected MMQ x-pipeline",
            owner="OPT-070",
            note="installed keep; evidence repair belongs to OPT-070",
            reevaluation={"status": "unresolved_dependency", "follow_up": "OPT-070"},
        )
    )
    entries.append(
        _performance(
            "OPT-066",
            "two_stage_x",
            "two-stage X prefetch is a resource no-go",
            "evidence/optimization/opt066-mmq-x-pipeline/REPORT.md",
            "none",
        )
    )
    entries.append(
        _base(
            "OPT-069",
            "combined_batch_gate",
            kind="blocked_gate",
            classification="blocked_gate",
            disposition="blocked",
            original_test="combined quality plus original P/D/2K",
            reference="same-sitting llama.cpp",
            metric="tok/s, p95, quality v2",
            envelope="OPT-056/016 unchanged",
            source="combined 062-068 selectors",
            latest="fixtures/opt069_batch_gate.json",
            owner="OPT-080",
            note="quality failures route through OPT-073; no numerical keep here",
        )
    )

    covered = {str(row["task"]) for row in entries}
    expected = {f"OPT-{index:03d}" for index in range(TASK_BEGIN, TASK_END + 1)}
    missing = sorted(expected - covered)
    if missing:
        raise RuntimeError(f"register missing tasks: {missing}")

    consumers = q8_1_consumer_matrix()
    return {
        "schema_version": 1,
        "task": "OPT-072",
        "status": "rejection_review",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "entry_count": len(entries),
        "tasks_covered": sorted(covered),
        "dispositions": list(DISPOSITIONS),
        "classifications": list(CLASSIFICATIONS),
        "kinds": list(KINDS),
        "entries": entries,
        "q8_1_consumer_audit": consumers,
        "independent_legacy_candidate": {
            "case_id": "half-scale-q8-1",
            "control": "packed Q8Block recomputes int32 sum(q)",
            "candidate": "integer UseQ81 consuming quartz half(sum(q))",
            "covered_by_opt075_079": False,
            "implementations": 2,
            "shapes": ["M17/K256", "M17408/K5120 sampled 16x4"],
            "correctness_executions": 1,
            "optional_screen": "1 warmup + 3 whole-family rounds",
            "screen_executed": False,
            "result_class": "enumerated_not_production_keep",
        },
        "follow_up_owners": {
            "OPT-070": "installed OPT-064/066 acceptance evidence",
            "OPT-073": "functional prompts/scorers and quality verdicts",
            "OPT-074": "production-shape llama GPU numerical admission",
            "OPT-075": "existing unfused/paired Q4 FFN keep/reject",
            "OPT-080": "combined original P/D/2K gates",
        },
    }


def q8_1_consumer_matrix() -> dict[str, Any]:
    cases = [integer_sum_audit(total) for total in INTEGER_SUM_CASES]
    scaled = integer_sum_audit(2047, scale=1.0 / 127.0)
    return {
        "sum_cases": list(INTEGER_SUM_CASES),
        "producer_quartz": {
            "id": STAGING_QUARTZ_SUM_Q,
            "stores": "half(sum(q))",
            "kernel": "quantize_bf16_q8_1",
        },
        "producer_llama": {
            "id": STAGING_LLAMA_SUM_X,
            "stores": "half(sum(x))",
            "kernel": "quantize_bf16_q8_1_sum_x",
        },
        "consumer_q4_useq81": {
            "id": STAGING_QUARTZ_SUM_Q,
            "reads": "rintf(half2float(q8_sum)) as int32 sum(q)",
            "recomputes_int32": False,
        },
        "consumer_q4_q8block": {
            "reads": "dp4a(0x01010101) exact int32 sum of quantized bytes",
            "recomputes_int32": True,
        },
        "consumer_q8_dp4a": {
            "reads": "scale and values only; stored sum unused",
            "production": True,
        },
        "consumer_q6_integer": {
            "reads": "scale and values only; stored sum unused",
            "production": True,
        },
        "illegal_pairing": {
            "producer": STAGING_LLAMA_SUM_X,
            "consumer": STAGING_QUARTZ_SUM_Q,
            "legal": False,
        },
        "cases": cases,
        "scaled_example_sum_x_differs": scaled["half_sum_x"] != scaled["half_sum_q"],
        "production_repair": (
            "typed pairing constants only; production Q8/Q6 staging unchanged"
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_register(path: Path = FIXTURE) -> dict[str, Any]:
    return load_json(path)


def covered_tasks(register: Mapping[str, Any]) -> set[str]:
    return {str(row["task"]) for row in register["entries"]}


def expected_tasks() -> set[str]:
    return {f"OPT-{index:03d}" for index in range(TASK_BEGIN, TASK_END + 1)}


def validate_register(register: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(expected_tasks() - covered_tasks(register))
    extra_disp = [
        row["id"]
        for row in register["entries"]
        if row.get("disposition") not in DISPOSITIONS
    ]
    extra_class = [
        row["id"]
        for row in register["entries"]
        if row.get("classification") not in CLASSIFICATIONS
    ]
    extra_kind = [
        row["id"] for row in register["entries"] if row.get("kind") not in KINDS
    ]
    incomplete = []
    for row in register["entries"]:
        reevaluation = row.get("reevaluation") or {}
        status = str(reevaluation.get("status", ""))
        owner = str(row.get("next_owner", ""))
        has_evidence = status in {
            "reproduced_from_retained_evidence",
            "proven_superseding_equivalence",
            "superseding_keep",
            "not_a_correctness_candidate",
            "performance_loss_not_rescued_by_tolerance",
            "historical_rule_retained_on_installed_keep",
            "unresolved_dependency",
        }
        if not has_evidence:
            incomplete.append(row["id"])
        if status == "unresolved_dependency" and owner in {"", "none"}:
            incomplete.append(row["id"])
        if not may_admit_by_numerical_headroom(row) and row.get(
            "admit_by_numerical_headroom"
        ):
            incomplete.append(row["id"])
    historical = []
    for row in register["entries"]:
        for rel in row.get("historical_files", []):
            path = ROOT / str(rel)
            if not path.is_file():
                historical.append(str(rel))
    ok = not missing and not extra_disp and not extra_class and not extra_kind
    ok = ok and not incomplete and not historical
    return {
        "success": ok,
        "missing_tasks": missing,
        "invalid_dispositions": extra_disp,
        "invalid_classifications": extra_class,
        "invalid_kinds": extra_kind,
        "incomplete_reevaluation": incomplete,
        "missing_historical_files": historical,
        "entry_count": len(register["entries"]),
    }


def validate_q8_1_audit(register: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audit = (
        q8_1_consumer_matrix() if register is None else register["q8_1_consumer_audit"]
    )
    by_int = {int(row["integer"]): row for row in audit["cases"]}
    failures: list[str] = []
    for total in INTEGER_SUM_CASES:
        row = by_int[int(total)]
        lost = integer_lost_in_half(int(total))
        if bool(row["units_lost_in_half"]) != lost:
            failures.append(f"{total} units_lost mismatch")
        if abs(total) == 2047 and lost:
            failures.append("2047 must be exact in half")
        if abs(total) == 2049 and not lost:
            failures.append("2049 must lose a unit in half")
        if abs(total) == 4064 and lost:
            failures.append("4064 must remain exact in half")
        if row["exact_int32_sum_q"] != int(total):
            failures.append(f"{total} exact int32 mismatch")
        if abs(total) <= 2048 and not row["sum_q_consumer_matches_exact"]:
            failures.append(f"{total} sum(q) consumer should match exact int32")
        if abs(total) == 2049 and row["sum_q_consumer_matches_exact"]:
            failures.append("2049 sum(q) half consumer must not match exact int32")
    if audit["illegal_pairing"]["legal"] is not False:
        failures.append("sum(x) producer to sum(q) consumer must be illegal")
    if not audit["scaled_example_sum_x_differs"]:
        failures.append("half(sum(x)) must differ from half(sum(q)) at non-unit scale")
    return {"success": not failures, "failures": failures, "cases": audit["cases"]}


def case_workloads(iteration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "product": loop_product(workload),
            "candidates": workload.get("candidates"),
            "cases": workload.get("cases"),
            "warmups": workload.get("warmups"),
            "samples": workload.get("samples"),
            "sampled_rows": workload.get("sampled_rows"),
            "sampled_vectors": workload.get("sampled_vectors"),
        }
        for name, workload in iteration.get("workloads", {}).items()
    }


def run_phase(phase: str, run_dir: Path | None = None) -> dict[str, Any]:
    register = load_register()
    iteration = load_json(ITERATION)
    contract = load_json(CONTRACT)
    if phase == "inventory":
        payload = validate_register(register)
        payload["result_class"] = "ok" if payload["success"] else "incomplete_inventory"
        payload["gpu_work"] = False
        payload["case_ids"] = list(iteration.get("case_ids", []))
        payload["workloads"] = case_workloads(iteration)
    elif phase == "q4-original-failure":
        q8 = validate_q8_1_audit(register)
        opt046 = next(
            row for row in register["entries"] if row["id"] == "OPT-046:integer_q8_w4"
        )
        payload = {
            "success": q8["success"],
            "result_class": "ok" if q8["success"] else "q8_1_audit_failed",
            "gpu_work": False,
            "original_failure": opt046["original_failure"],
            "v2_comparison": opt046["v2_comparison"],
            "q8_1_audit": q8,
            "native_targets": opt046.get("native_targets", []),
            "route": ["OPT-074", "OPT-075"],
            "historical_30_sample_oracle": False,
        }
    elif phase == "review":
        inventory = validate_register(register)
        q8 = validate_q8_1_audit(register)
        blocked = [
            {
                "id": row["id"],
                "owner": row["next_owner"],
                "disposition": row["disposition"],
            }
            for row in register["entries"]
            if row["disposition"] in {"blocked", "numerically_eligible"}
        ]
        complete = inventory["success"] and q8["success"]
        payload = {
            "success": complete,
            "result_class": "ok" if complete else "incomplete_review",
            "gpu_work": False,
            "inventory": inventory,
            "q8_1_audit": {"success": q8["success"], "failures": q8["failures"]},
            "blocked_or_eligible": blocked,
            "independent_legacy_candidate": register["independent_legacy_candidate"],
            "claims_throughput": False,
            "acceptance": "review_completeness",
            "contract_task": contract["task"],
        }
    else:
        raise ValueError(f"unknown OPT-072 phase {phase}")
    payload["task"] = "OPT-072"
    payload["phase"] = phase
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt072-result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def write_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    register = build_register()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, indent=2) + "\n", encoding="utf-8")
    return register


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("inventory", "q4-original-failure", "review", "write-fixture"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == "write-fixture":
        register = write_fixture()
        print(json.dumps({"success": True, "entry_count": register["entry_count"]}))
        return 0
    result = run_phase(args.phase, args.run_dir)
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
