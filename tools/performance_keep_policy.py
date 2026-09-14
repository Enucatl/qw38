"""Host-only target_guard_v2 keep policy: validation and statistics.

Prospective internal admission for tasks that opt in. Does not rewrite
historical fixtures, production pins, or release gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLICY_PATH = ROOT / "pins/performance_keep_policy_v2.json"
POLICY_ID = "target_guard_v2"
OPT130_FIXTURE = ROOT / "fixtures/opt130_dense_attention.json"
FREEZE_KEYS = (
    "policy_id",
    "candidate_id",
    "targets",
    "guards",
    "dispatch_region",
    "unchanged_branches",
    "quality_gates",
    "state_gates",
    "thresholds",
)
IDENTITY_FIELDS = (
    "prefix",
    "eval_count",
    "capacity",
    "metric_boundary",
    "sampling_policy",
    "output_policy",
    "engine_config",
    "input_trajectory",
)
DECODE_METRICS = frozenset({"decode_only", "complete_request"})
PREFILL_METRICS = frozenset({"prefill"})
KNOWN_METRICS = DECODE_METRICS | PREFILL_METRICS
VERDICT_RANK = {
    "incomplete": 0,
    "reject": 1,
    "inconclusive": 2,
    "keep": 3,
}


class KeepPolicyError(ValueError):
    """Unknown policy ID, schema error, or frozen-hash mismatch."""


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KeepPolicyError(f"{path} is not a JSON object")
    return payload


def load_policy(path: Path | None = None) -> dict[str, Any]:
    payload = load_json(path or POLICY_PATH)
    policy_id = str(payload.get("policy_id") or "")
    if policy_id != POLICY_ID:
        raise KeepPolicyError(f"unknown keep policy id {policy_id!r}; fail closed")
    return payload


def policy_id_of(contract: Mapping[str, Any]) -> str | None:
    for key in ("keep_policy_id", "performance_keep_policy_id", "policy_id"):
        value = contract.get(key)
        if value:
            return str(value)
    nested = contract.get("keep_policy")
    if isinstance(nested, Mapping):
        value = nested.get("id") or nested.get("policy_id")
        if value:
            return str(value)
    return None


def candidate_contract_view(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = contract.get("keep_policy")
    if isinstance(nested, Mapping) and (
        "targets" in nested or "guards" in nested or "policy_id" in nested
    ):
        merged = dict(nested)
        if "policy_id" not in merged:
            merged["policy_id"] = policy_id_of(contract)
        return merged
    return contract


def freeze_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    view = candidate_contract_view(contract)
    payload = {key: view.get(key) for key in FREEZE_KEYS if key in view}
    payload.setdefault("policy_id", policy_id_of(contract))
    return payload


def freeze_hash(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        freeze_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_workloads(rows: Any, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise KeepPolicyError(f"{field} must be a nonempty list")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not str(row.get("workload") or ""):
            raise KeepPolicyError(f"{field} entries need a workload id")
        out.append(dict(row))
    return out


def _kind_of(row: Mapping[str, Any], *, default: str | None = None) -> str:
    kind = str(row.get("kind") or default or "")
    if kind not in {"decode", "prefill"}:
        primary = str(row.get("primary_metric") or "")
        metrics = row.get("metrics") or []
        if primary == "prefill" or metrics == ["prefill"]:
            kind = "prefill"
        elif primary in DECODE_METRICS or any(
            str(item) in DECODE_METRICS for item in metrics
        ):
            kind = "decode"
    if kind not in {"decode", "prefill"}:
        raise KeepPolicyError(
            f"workload {row.get('workload')!r} needs kind decode or prefill"
        )
    return kind


def required_roles(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    view = candidate_contract_view(contract)
    targets = _as_workloads(view.get("targets"), field="targets")
    guards = _as_workloads(view.get("guards"), field="guards")
    roles: dict[str, dict[str, Any]] = {}
    for row in targets:
        name = str(row["workload"])
        if name in roles:
            raise KeepPolicyError(f"workload {name} has multiple roles")
        kind = _kind_of(row)
        primary = str(row.get("primary_metric") or "")
        if kind == "decode":
            if primary and primary != "decode_only":
                raise KeepPolicyError(
                    f"decode target {name} primary_metric must be decode_only"
                )
            primary = "decode_only"
            metrics = ["decode_only", "complete_request"]
        else:
            if primary and primary != "prefill":
                raise KeepPolicyError(
                    f"prefill target {name} primary_metric must be prefill"
                )
            primary = "prefill"
            metrics = ["prefill"]
        roles[name] = {
            "role": "target",
            "kind": kind,
            "primary_metric": primary,
            "metrics": metrics,
            "row": row,
        }
    for row in guards:
        name = str(row["workload"])
        if name in roles:
            raise KeepPolicyError(f"targets and guards are not disjoint ({name})")
        kind = _kind_of(row)
        listed = [str(item) for item in (row.get("metrics") or [])]
        if listed:
            metrics = listed
        elif kind == "decode":
            metrics = ["decode_only", "complete_request"]
        else:
            metrics = ["prefill"]
        unknown = [item for item in metrics if item not in KNOWN_METRICS]
        if unknown:
            raise KeepPolicyError(f"unknown metrics {unknown} on {name}")
        roles[name] = {
            "role": "guard",
            "kind": kind,
            "primary_metric": None,
            "metrics": metrics,
            "row": row,
        }
    return roles


def validate_candidate_schema(contract: Mapping[str, Any]) -> None:
    policy = policy_id_of(contract)
    if policy is None:
        return
    if policy != POLICY_ID:
        raise KeepPolicyError(f"unknown keep policy id {policy!r}; fail closed")
    view = candidate_contract_view(contract)
    if "targets" not in view and "guards" not in view:
        return
    required_roles(contract)
    quality = view.get("quality_gates")
    state = view.get("state_gates")
    if not isinstance(quality, list) or not quality:
        raise KeepPolicyError("quality_gates must be a nonempty list")
    if not isinstance(state, list) or not state:
        raise KeepPolicyError("state_gates must be a nonempty list")
    if not str(view.get("dispatch_region") or "").strip():
        raise KeepPolicyError("dispatch_region is required")
    if not isinstance(view.get("unchanged_branches"), list):
        raise KeepPolicyError("unchanged_branches must be a list")
    thresholds = view.get("thresholds")
    if thresholds is not None and not isinstance(thresholds, Mapping):
        raise KeepPolicyError("thresholds must be an object")


def validate_opt_in_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed on unknown IDs; schema-check only when roles are present."""
    policy = policy_id_of(contract)
    if policy is None:
        return
    if policy != POLICY_ID:
        raise KeepPolicyError(f"unknown keep policy id {policy!r}; fail closed")
    view = candidate_contract_view(contract)
    if any(key in view for key in ("targets", "guards")):
        validate_candidate_schema(contract)


def log_interval(
    ratios: Sequence[float], *, t_critical: float, n_pairs: int
) -> dict[str, Any]:
    if len(ratios) != n_pairs:
        return {
            "ok": False,
            "reason": "acceptance_pairs_must_be_10",
            "n": len(ratios),
        }
    xs: list[float] = []
    for ratio in ratios:
        value = float(ratio)
        if not math.isfinite(value) or value <= 0.0:
            return {
                "ok": False,
                "reason": "non_finite_or_non_positive_ratio",
            }
        xs.append(math.log(value))
    mean_x = sum(xs) / float(len(xs))
    g = math.exp(mean_x)
    var = (
        sum((item - mean_x) ** 2 for item in xs) / float(len(xs) - 1)
        if len(xs) > 1
        else 0.0
    )
    if var <= 0.0:
        return {
            "ok": True,
            "g": g,
            "se": 0.0,
            "L": g,
            "U": g,
            "mean_x": mean_x,
            "n": len(xs),
        }
    se = math.sqrt(var / float(len(xs)))
    return {
        "ok": True,
        "g": g,
        "se": se,
        "L": math.exp(mean_x - t_critical * se),
        "U": math.exp(mean_x + t_critical * se),
        "mean_x": mean_x,
        "n": len(xs),
    }


def _gate_pass(mapping: Any, name: str) -> str:
    if not isinstance(mapping, Mapping) or name not in mapping:
        return "incomplete"
    value = mapping[name]
    if value is True or value == "pass" or value == 1:
        return "pass"
    if value is False or value == "fail" or value == 0:
        return "reject"
    if isinstance(value, Mapping):
        if value.get("pass") is True or value.get("ok") is True:
            return "pass"
        if value.get("pass") is False or value.get("ok") is False:
            return "reject"
        return "incomplete"
    return "incomplete"


def _identity_key(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row.get("identity")
    source = dict(identity) if isinstance(identity, Mapping) else {}
    for field in IDENTITY_FIELDS:
        if field in row and field not in source:
            source[field] = row[field]
    source.setdefault("metric_boundary", row.get("metric"))
    return {field: source.get(field) for field in IDENTITY_FIELDS}


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def pairs_from_ratios(
    *,
    workload: str,
    metric: str,
    ratios: Sequence[float],
    p95_ratios: Sequence[float] | None = None,
    control_rate: float = 100.0,
    control_p95: float = 10.0,
    identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p95 = list(p95_ratios) if p95_ratios is not None else [1.0] * len(ratios)
    base = {
        "prefix": 2048,
        "eval_count": 256,
        "capacity": 4096,
        "metric_boundary": metric,
        "sampling_policy": "greedy",
        "output_policy": "fixed_eval",
        "engine_config": "quartz_graph",
        "input_trajectory": "deterministic",
    }
    if identity:
        base.update(dict(identity))
    base["metric_boundary"] = metric
    for index, ratio in enumerate(ratios):
        p95_ratio = float(p95[index]) if index < len(p95) else 1.0
        rows.append(
            {
                "sample_id": index,
                "order": "AB" if index % 2 == 0 else "BA",
                "workload": workload,
                "metric": metric,
                "control_rate": float(control_rate),
                "candidate_rate": float(control_rate) * float(ratio),
                "control_p95_itl": float(control_p95),
                "candidate_p95_itl": float(control_p95) * p95_ratio,
                "identity": dict(base),
            }
        )
    return rows


def default_candidate_contract(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "candidate_id": "opt135_synthetic",
        "dispatch_region": "decode>=2048",
        "unchanged_branches": ["decode<2048", "prefill"],
        "quality_gates": ["quality"],
        "state_gates": ["state"],
        "targets": [
            {
                "workload": "d2048",
                "kind": "decode",
                "primary_metric": "decode_only",
            }
        ],
        "guards": [
            {"workload": "d128", "kind": "decode"},
            {"workload": "p4096", "kind": "prefill"},
        ],
    }
    payload.update(overrides)
    return payload


def complete_records(
    *,
    target_decode: Sequence[float],
    target_request: Sequence[float] | None = None,
    guard_decode: Sequence[float] | None = None,
    guard_request: Sequence[float] | None = None,
    prefill: Sequence[float] | None = None,
    target_p95: Sequence[float] | None = None,
    extra_pairs: Sequence[Mapping[str, Any]] | None = None,
    quality: bool | None = True,
    state: bool | None = True,
    extra_targets: (
        Sequence[tuple[str, Sequence[float], Sequence[float]]] | None
    ) = None,
    **flags: Any,
) -> dict[str, Any]:
    ones = [1.0] * len(target_decode)
    request = list(target_request if target_request is not None else ones)
    g_decode = list(guard_decode if guard_decode is not None else ones)
    g_request = list(guard_request if guard_request is not None else ones)
    prefill_ratios = list(prefill if prefill is not None else ones)
    pairs: list[dict[str, Any]] = []
    pairs.extend(
        pairs_from_ratios(
            workload="d2048",
            metric="decode_only",
            ratios=target_decode,
            p95_ratios=target_p95,
            identity={"prefix": 2048},
        )
    )
    pairs.extend(
        pairs_from_ratios(
            workload="d2048",
            metric="complete_request",
            ratios=request,
            p95_ratios=target_p95,
            identity={"prefix": 2048},
        )
    )
    pairs.extend(
        pairs_from_ratios(
            workload="d128",
            metric="decode_only",
            ratios=g_decode,
            identity={"prefix": 128, "eval_count": 256},
        )
    )
    pairs.extend(
        pairs_from_ratios(
            workload="d128",
            metric="complete_request",
            ratios=g_request,
            identity={"prefix": 128, "eval_count": 256},
        )
    )
    pairs.extend(
        pairs_from_ratios(
            workload="p4096",
            metric="prefill",
            ratios=prefill_ratios,
            p95_ratios=[0.0] * len(prefill_ratios),
            identity={
                "prefix": 4096,
                "eval_count": 4096,
                "metric_boundary": "prefill",
            },
        )
    )
    if extra_targets:
        for workload, decode_ratios, request_ratios in extra_targets:
            digits = workload[1:] if workload[:1] in {"d", "p"} else ""
            prefix = int(digits) if digits.isdigit() else 0
            pairs.extend(
                pairs_from_ratios(
                    workload=workload,
                    metric="decode_only",
                    ratios=decode_ratios,
                    identity={"prefix": prefix},
                )
            )
            pairs.extend(
                pairs_from_ratios(
                    workload=workload,
                    metric="complete_request",
                    ratios=request_ratios,
                    identity={"prefix": prefix},
                )
            )
    if extra_pairs:
        pairs.extend(dict(row) for row in extra_pairs)
    payload: dict[str, Any] = {"pairs": pairs}
    if quality is not None:
        payload["quality"] = {"quality": quality}
    if state is not None:
        payload["state"] = {"state": state}
    payload.update(flags)
    return payload


def records_from_opt130(path: Path | None = None) -> dict[str, Any]:
    payload = load_json(path or OPT130_FIXTURE)
    performance = payload.get("performance") or {}
    pairs: list[dict[str, Any]] = []
    identities = {
        "d128": {
            "prefix": 128,
            "eval_count": 256,
            "capacity": 131072,
            "sampling_policy": "greedy",
            "output_policy": "fixed_eval",
            "engine_config": "quartz_graph",
            "input_trajectory": "opt130_dense_attention",
        },
        "d2048": {
            "prefix": 2048,
            "eval_count": 256,
            "capacity": 131072,
            "sampling_policy": "greedy",
            "output_policy": "fixed_eval",
            "engine_config": "quartz_graph",
            "input_trajectory": "opt130_dense_attention",
        },
        "p4096": {
            "prefix": 4096,
            "eval_count": 4096,
            "capacity": 131072,
            "sampling_policy": "greedy",
            "output_policy": "fixed_eval",
            "engine_config": "quartz_graph",
            "input_trajectory": "opt130_dense_attention",
        },
    }
    metric_map = {
        "decode_only": "decode_only",
        "request": "complete_request",
    }
    for workload, block in performance.items():
        if not isinstance(block, Mapping):
            continue
        for source_name, metric in metric_map.items():
            family = block.get(source_name)
            if not isinstance(family, Mapping):
                continue
            recorded = family.get("pairs") or []
            identity = dict(identities.get(workload, {}))
            use_metric = "prefill" if workload == "p4096" else metric
            if workload == "p4096" and source_name != "request":
                continue
            identity["metric_boundary"] = use_metric
            for index, pair in enumerate(recorded):
                if not isinstance(pair, Mapping):
                    continue
                control = pair.get("A") if isinstance(pair.get("A"), Mapping) else {}
                candidate = pair.get("B") if isinstance(pair.get("B"), Mapping) else {}
                if use_metric in {"complete_request", "prefill"}:
                    control_rate = control.get("request_tok_s") or control.get("tok_s")
                    candidate_rate = candidate.get("request_tok_s") or candidate.get(
                        "tok_s"
                    )
                else:
                    control_rate = control.get("decode_only_tok_s") or control.get(
                        "tok_s"
                    )
                    candidate_rate = candidate.get(
                        "decode_only_tok_s"
                    ) or candidate.get("tok_s")
                pairs.append(
                    {
                        "sample_id": pair.get("sample_index", index),
                        "order": pair.get("order")
                        or ("AB" if index % 2 == 0 else "BA"),
                        "workload": workload,
                        "metric": use_metric,
                        "control_rate": control_rate,
                        "candidate_rate": candidate_rate,
                        "control_p95_itl": control.get("p95_ms"),
                        "candidate_p95_itl": candidate.get("p95_ms"),
                        "identity": dict(identity),
                    }
                )
    quality = (
        payload.get("quality") if isinstance(payload.get("quality"), Mapping) else {}
    )
    state = (
        payload.get("state_memory")
        if isinstance(payload.get("state_memory"), Mapping)
        else {}
    )
    return {
        "pairs": pairs,
        "quality": {"quality": bool(quality.get("ok"))} if "ok" in quality else {},
        "state": {"state": bool(state.get("ok"))} if "ok" in state else {},
        "historical_verdict": payload.get("verdict"),
        "counterfactual": True,
        "source": "fixtures/opt130_dense_attention.json",
    }


def _group_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pairs:
        workload = str(row.get("workload") or "")
        metric = str(row.get("metric") or "")
        grouped.setdefault((workload, metric), []).append(dict(row))
    return grouped


def _validate_group(rows: Sequence[Mapping[str, Any]], *, n_pairs: int) -> list[str]:
    reasons: list[str] = []
    if len(rows) != n_pairs:
        reasons.append("acceptance_pairs_must_be_10")
    sample_ids: list[Any] = []
    identities: list[dict[str, Any]] = []
    expected_orders = ["AB" if index % 2 == 0 else "BA" for index in range(len(rows))]
    orders = [str(row.get("order") or "") for row in rows]
    if orders != expected_orders:
        reasons.append("pairs_must_alternate_ab_ba")
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is None:
            reasons.append("missing_sample_id")
        sample_ids.append(sample_id)
        if not _finite_positive(row.get("control_rate")) or not _finite_positive(
            row.get("candidate_rate")
        ):
            reasons.append("non_finite_or_non_positive_rate")
        identities.append(_identity_key(row))
        metric = str(row.get("metric") or "")
        boundary = identities[-1].get("metric_boundary")
        if boundary not in (None, metric):
            reasons.append("mixed_metric_identity")
    if len(set(sample_ids)) != len(sample_ids):
        reasons.append("duplicate_sample_id")
    if identities and any(item != identities[0] for item in identities[1:]):
        reasons.append("changed_identity")
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return unique


def _p95_decision(
    rows: Sequence[Mapping[str, Any]], *, limit: float, kind: str
) -> tuple[str, list[float | None], list[str]]:
    ratios: list[float | None] = []
    reasons: list[str] = []
    if kind != "decode":
        for row in rows:
            control = row.get("control_p95_itl")
            candidate = row.get("candidate_p95_itl")
            if (
                control in (None, 0, 0.0)
                or candidate in (None, 0, 0.0)
                or not _finite_positive(control)
            ):
                ratios.append(None)
            else:
                ratios.append(float(candidate) / float(control))
        return "pass", ratios, reasons
    for index, row in enumerate(rows):
        control = row.get("control_p95_itl")
        candidate = row.get("candidate_p95_itl")
        if not _finite_positive(control) or not _finite_positive(candidate):
            ratios.append(None)
            reasons.append(f"missing_or_zero_p95_pair_{index}")
            continue
        ratio = float(candidate) / float(control)
        ratios.append(ratio)
        if ratio > limit:
            reasons.append(f"p95_itl_ratio_{index}>{limit}")
    if any(item is None for item in ratios):
        return "incomplete", ratios, reasons
    if any(reason.startswith("p95_itl_ratio_") for reason in reasons):
        return "reject", ratios, reasons
    return "pass", ratios, reasons


def _as_verdict(status: str) -> str:
    if status in {"pass", "keep"}:
        return "keep"
    if status in VERDICT_RANK:
        return status
    return "incomplete"


METRIC_RANK = {
    "incomplete": 0,
    "reject": 1,
    "inconclusive": 2,
    "pass": 3,
}


def _combine_metric(statuses: Iterable[str]) -> str:
    mapped = ["pass" if item == "keep" else item for item in statuses]
    rank = min((METRIC_RANK.get(item, 0) for item in mapped), default=0)
    for name, value in METRIC_RANK.items():
        if value == rank:
            return name
    return "incomplete"


def _combine(statuses: Iterable[str]) -> str:
    mapped = [_as_verdict(item) for item in statuses]
    rank = min((VERDICT_RANK.get(item, 0) for item in mapped), default=0)
    for name, value in VERDICT_RANK.items():
        if value == rank:
            return name
    return "incomplete"


def _metric_verdict(
    interval: Mapping[str, Any], *, role: str, guard_floor: float
) -> str:
    if not interval.get("ok"):
        return "incomplete"
    lower = float(interval["L"])
    upper = float(interval["U"])
    if role == "target":
        if upper <= 1.0:
            return "reject"
        if lower > 1.0:
            return "pass"
        return "inconclusive"
    if upper < guard_floor:
        return "reject"
    if lower >= guard_floor:
        return "pass"
    return "inconclusive"


def evaluate(
    contract: Mapping[str, Any],
    records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate acceptance pairs under target_guard_v2.

    Returns a verdict of keep/reject/inconclusive/incomplete. Unknown policy
    IDs and frozen-hash mismatches fail closed via KeepPolicyError.
    """
    policy_payload = policy or load_policy()
    if str(policy_payload.get("policy_id") or "") != POLICY_ID:
        raise KeepPolicyError(
            f"unknown keep policy id {policy_payload.get('policy_id')!r}; fail closed"
        )
    declared = policy_id_of(contract)
    if declared is None:
        raise KeepPolicyError("keep policy id is required; fail closed")
    if declared != POLICY_ID:
        raise KeepPolicyError(f"unknown keep policy id {declared!r}; fail closed")
    validate_candidate_schema(contract)
    view = candidate_contract_view(contract)
    thresholds = dict(policy_payload)
    extra = view.get("thresholds")
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            expected = policy_payload.get(key)
            if expected is not None and value != expected:
                raise KeepPolicyError(
                    f"post-hoc threshold change {key}={value!r}; fail closed"
                )
        thresholds.update(dict(extra))
    n_pairs = int(thresholds.get("acceptance_pairs") or 10)
    t_critical = float(thresholds.get("t_critical_df9_one_sided_95") or 0.0)
    guard_floor = float(thresholds.get("guard_l_min") or 0.98)
    p95_limit = float(thresholds.get("p95_itl_ratio_max") or 1.05)
    material_threshold = float(thresholds.get("material_2pct_threshold") or 1.02)
    if n_pairs != 10:
        raise KeepPolicyError("this policy version requires exactly ten pairs")

    payload: dict[str, Any]
    if isinstance(records, Mapping):
        payload = dict(records)
    else:
        payload = {"pairs": list(records)}
    computed_hash = freeze_hash(contract)
    declared_hash = view.get("contract_hash") or payload.get("contract_hash")
    if declared_hash is not None and str(declared_hash) != computed_hash:
        raise KeepPolicyError("frozen-role hash mismatch; fail closed")

    pairs = list(payload.get("pairs") or [])
    grouped = _group_pairs(pairs)
    roles = required_roles(contract)
    reasons: list[str] = []
    statuses: list[str] = []
    metrics: dict[str, Any] = {}
    candidate_rates: dict[str, Any] = {}

    quality_gates = [str(item) for item in (view.get("quality_gates") or [])]
    state_gates = [str(item) for item in (view.get("state_gates") or [])]
    quality = payload.get("quality")
    state = payload.get("state")
    for name in quality_gates:
        gate = _gate_pass(quality, name)
        if gate == "incomplete":
            reasons.append(f"missing_quality:{name}")
            statuses.append("incomplete")
        elif gate == "reject":
            reasons.append(f"quality_fail:{name}")
            statuses.append("reject")
        else:
            statuses.append("keep")
    for name in state_gates:
        gate = _gate_pass(state, name)
        if gate == "incomplete":
            reasons.append(f"missing_state:{name}")
            statuses.append("incomplete")
        elif gate == "reject":
            reasons.append(f"state_fail:{name}")
            statuses.append("reject")
        else:
            statuses.append("keep")

    for workload, spec in roles.items():
        kind = str(spec["kind"])
        p95_checked = False
        for metric in spec["metrics"]:
            rows = grouped.get((workload, metric), [])
            key = f"{workload}.{metric}"
            group_reasons = _validate_group(rows, n_pairs=n_pairs)
            role = (
                "target"
                if spec["role"] == "target" and metric == spec["primary_metric"]
                else "guard"
            )
            if not rows:
                group_reasons.append("missing_pairs")
            ratios = []
            for row in rows:
                control = float(row.get("control_rate") or 0.0)
                candidate = float(row.get("candidate_rate") or 0.0)
                ratios.append(candidate / control if control > 0.0 else 0.0)
            interval = (
                log_interval(ratios, t_critical=t_critical, n_pairs=n_pairs)
                if not group_reasons
                else {"ok": False, "reason": group_reasons[0]}
            )
            metric_status = (
                "incomplete"
                if group_reasons
                else _metric_verdict(interval, role=role, guard_floor=guard_floor)
            )
            p95_status = "pass"
            p95_ratios: list[float | None] = []
            p95_reasons: list[str] = []
            if kind == "decode" and rows and not p95_checked:
                p95_status, p95_ratios, p95_reasons = _p95_decision(
                    rows, limit=p95_limit, kind=kind
                )
                p95_checked = True
            reasons.extend(f"{key}:{item}" for item in group_reasons)
            reasons.extend(f"{workload}:{item}" for item in p95_reasons)
            metric_status = _combine_metric([metric_status, p95_status])
            statuses.append(metric_status)
            g_value = interval.get("g") if interval.get("ok") else None
            metrics[key] = {
                "workload": workload,
                "metric": metric,
                "role": role,
                "kind": kind,
                "g": g_value,
                "L": interval.get("L") if interval.get("ok") else None,
                "U": interval.get("U") if interval.get("ok") else None,
                "se": interval.get("se") if interval.get("ok") else None,
                "material_2pct": (
                    bool(g_value is not None and float(g_value) >= material_threshold)
                    if role == "target" and g_value is not None
                    else None
                ),
                "p95_ratios": p95_ratios,
                "decision": metric_status if metric_status != "pass" else "pass",
                "reasons": group_reasons + p95_reasons,
            }
            if role == "target" and interval.get("ok"):
                mean_control = sum(float(row["control_rate"]) for row in rows) / float(
                    len(rows)
                )
                mean_candidate = sum(
                    float(row["candidate_rate"]) for row in rows
                ) / float(len(rows))
                candidate_rates[key] = {
                    "control_rate": mean_control,
                    "candidate_rate": mean_candidate,
                    "delta": mean_candidate - mean_control,
                    "g": g_value,
                }

    if payload.get("phase") == "screen":
        reasons.append("screen_never_admits_keep")
        statuses.append("incomplete")

    verdict = _combine(statuses or ["incomplete"])

    unique_reasons: list[str] = []
    seen_reason: set[str] = set()
    for item in reasons:
        if item not in seen_reason:
            seen_reason.add(item)
            unique_reasons.append(item)

    historical = payload.get("historical_verdict")
    counterfactual = bool(payload.get("counterfactual") or view.get("counterfactual"))
    if counterfactual:
        unique_reasons.append(
            "counterfactual_replay_does_not_alter_historical_admission"
        )
        if historical:
            verdict = str(historical)
        elif verdict == "keep":
            verdict = "reject"

    return {
        "policy_id": POLICY_ID,
        "verdict": verdict,
        "reasons": unique_reasons,
        "metrics": metrics,
        "candidate_rates": candidate_rates,
        "shipping_impact": 0,
        "counterfactual": counterfactual,
        "historical_verdict": historical,
        "contract_hash": computed_hash,
        "interval_label": "one_sided_95_bounds",
    }


def self_check(
    fixture_path: Path, *, policy_path: Path | None = None
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    fixture = load_json(fixture_path)
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise KeepPolicyError(f"{fixture_path} has no cases")
    reports: list[dict[str, Any]] = []
    mismatches = 0
    for case in cases:
        case_id = str(case.get("id") or "unnamed")
        contract = case.get("contract") or {}
        records = dict(case.get("records") or {})
        source = case.get("source")
        expected = case.get("expected") or {}
        if source == "opt130":
            loaded = records_from_opt130()
            loaded.update({key: records[key] for key in records})
            records = loaded
        row: dict[str, Any] = {"id": case_id}
        try:
            result = evaluate(contract, records, policy=policy)
        except KeepPolicyError as exc:
            failed_closed = True
            message = str(exc)
            result = None
        else:
            failed_closed = False
            message = ""
        want_closed = bool(expected.get("fail_closed"))
        matched = failed_closed == want_closed
        if want_closed:
            token = str(expected.get("error") or "")
            if token and token not in message:
                matched = False
        elif result is not None:
            if result.get("verdict") != expected.get("verdict"):
                matched = False
            wanted_metrics = expected.get("metrics") or {}
            if isinstance(wanted_metrics, Mapping):
                for key, decision in wanted_metrics.items():
                    actual = (result.get("metrics") or {}).get(key, {})
                    if actual.get("decision") != decision:
                        matched = False
            if "historical_verdict" in expected and result.get(
                "historical_verdict"
            ) != expected.get("historical_verdict"):
                matched = False
            wanted_material = expected.get("material_2pct")
            if isinstance(wanted_material, Mapping):
                for key, flag in wanted_material.items():
                    actual = (result.get("metrics") or {}).get(key, {})
                    if actual.get("material_2pct") != flag:
                        matched = False
        if not matched:
            mismatches += 1
        row.update(
            {
                "matched": matched,
                "fail_closed": failed_closed,
                "error": message,
                "verdict": None if result is None else result.get("verdict"),
                "expected_verdict": expected.get("verdict"),
            }
        )
        reports.append(row)
        status = "ok" if matched else "mismatch"
        print(f"{status}\t{case_id}\t{row.get('verdict') or row.get('error')}")
    return {
        "cases": reports,
        "mismatches": mismatches,
        "ok": mismatches == 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=POLICY_PATH,
        help="Frozen policy pin (pins/performance_keep_policy_v2.json)",
    )
    parser.add_argument(
        "--self-check",
        type=Path,
        default=None,
        help="Fixture of candidate contracts and raw paired records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_policy(args.contract)
        if args.self_check is None:
            print(f"policy_id={POLICY_ID}")
            return 0
        result = self_check(args.self_check, policy_path=args.contract)
    except KeepPolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
