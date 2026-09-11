"""Stream-aware family aggregation for OPT-060 engine attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
PATCH = ROOT / "tools/llama_authority/patches/opt060-engine-attribution.patch"
OVERLAY = ROOT / "tools/llama_authority/patches/qw38_opt060_attribution.cuh"
REQUIRED_ROLES = (
    "embedding",
    "ffn_gate_up_glu",
    "ffn_down",
    "logits_projection",
    "activation_quant",
    "copy",
    "cpu_unknown_gap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def patch_identity() -> dict[str, str]:
    return {
        "revision": LLAMA_REV,
        "patch": str(PATCH.relative_to(ROOT)),
        "patch_sha256": sha256_file(PATCH) if PATCH.is_file() else "",
        "overlay": str(OVERLAY.relative_to(ROOT)),
        "overlay_sha256": sha256_file(OVERLAY) if OVERLAY.is_file() else "",
    }


def _ms(record: Mapping[str, Any], key: str = "complete_work_ms") -> float:
    return float(record.get(key, 0.0) or 0.0)


def interval(record: Mapping[str, Any]) -> tuple[float, float]:
    start = float(record.get("start_ms", 0.0) or 0.0)
    end = float(record.get("end_ms", start + _ms(record)))
    if end < start:
        end = start
    return start, end


def union_ms(records: Sequence[Mapping[str, Any]]) -> float:
    spans = sorted(interval(record) for record in records)
    if not spans:
        return 0.0
    total = 0.0
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def summed_work_ms(records: Sequence[Mapping[str, Any]]) -> float:
    return sum(_ms(record) for record in records)


def stream_aware_totals(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stream: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = str(record.get("stream_index", record.get("stream", "0")))
        by_stream.setdefault(key, []).append(record)
    stream_sums = {key: summed_work_ms(group) for key, group in by_stream.items()}
    work = sum(stream_sums.values())
    wall = union_ms(records)
    return {
        "summed_gpu_work_ms": work,
        "interval_union_ms": wall,
        "overlap_ms": max(0.0, work - wall),
        "streams": stream_sums,
        "sum_exceeds_wall": work > wall + 1e-9,
    }


def _member_names(record: Mapping[str, Any]) -> set[str]:
    raw = str(record.get("fused_member_ids", "") or "")
    if not raw.strip():
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def chargeable_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep enclosing fused intervals once; drop members already charged."""
    enclosing = [
        record
        for record in records
        if str(record.get("attribution_role", "")) == "enclosing"
        and int(record.get("fused_member_count", 1) or 1) > 1
    ]
    charged_layers: dict[tuple[Any, Any], set[str]] = {}
    for record in enclosing:
        key = (record.get("engine"), record.get("layer"), record.get("phase"))
        charged_layers.setdefault(key, set()).update(_member_names(record))
        charged_layers[key].add(str(record.get("role", "")))
    kept: list[Mapping[str, Any]] = []
    for record in records:
        role = str(record.get("role", ""))
        key = (record.get("engine"), record.get("layer"), record.get("phase"))
        members = charged_layers.get(key, set())
        if str(record.get("attribution_role", "")) == "member" and (
            role in members or bool(set(role.split(",")) & members)
        ):
            continue
        kept.append(record)
    return kept


def family_table(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    charged = chargeable_records(records)
    families: dict[str, dict[str, Any]] = {}
    for record in charged:
        name = str(record.get("role") or "unknown")
        bucket = families.setdefault(
            name,
            {
                "role": name,
                "ms": 0.0,
                "calls": 0,
                "fused_member_count": 0,
                "engines": set(),
                "launch_families": set(),
            },
        )
        bucket["ms"] += _ms(record)
        bucket["calls"] += 1
        bucket["fused_member_count"] = max(
            int(bucket["fused_member_count"]),
            int(record.get("fused_member_count", 1) or 1),
        )
        bucket["engines"].add(str(record.get("engine", "")))
        bucket["launch_families"].add(str(record.get("launch_family", "")))
    out: dict[str, dict[str, Any]] = {}
    for name, bucket in families.items():
        out[name] = {
            "role": name,
            "ms": bucket["ms"],
            "calls": bucket["calls"],
            "fused_member_count": bucket["fused_member_count"],
            "engines": sorted(bucket["engines"]),
            "launch_families": sorted(bucket["launch_families"]),
        }
    return out


def missing_roles(
    records: Sequence[Mapping[str, Any]],
    required: Iterable[str] = REQUIRED_ROLES,
) -> list[str]:
    present = {str(record.get("role", "")) for record in records}
    for record in records:
        present.update(_member_names(record))
    return [role for role in required if role not in present]


def unmatched_roles(
    quartz: Sequence[Mapping[str, Any]],
    llama: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    left = {str(record.get("role", "")) for record in quartz}
    right = {str(record.get("role", "")) for record in llama}
    return {
        "quartz_only": sorted(left - right),
        "llama_only": sorted(right - left),
    }


def compare_engines(
    quartz: Sequence[Mapping[str, Any]],
    llama: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    qtab = family_table(quartz)
    ltab = family_table(llama)
    names = sorted(set(qtab) | set(ltab))
    rows = []
    for name in names:
        qms = float(qtab.get(name, {}).get("ms", 0.0))
        lms = float(ltab.get(name, {}).get("ms", 0.0))
        rows.append(
            {
                "role": name,
                "quartz_ms": qms,
                "llama_ms": lms,
                "quartz_minus_llama_ms": qms - lms,
                "quartz_calls": int(qtab.get(name, {}).get("calls", 0)),
                "llama_calls": int(ltab.get(name, {}).get("calls", 0)),
            }
        )
    q_tot = stream_aware_totals(chargeable_records(quartz))
    l_tot = stream_aware_totals(chargeable_records(llama))
    return {
        "families": rows,
        "quartz_work_ms": q_tot["summed_gpu_work_ms"],
        "llama_work_ms": l_tot["summed_gpu_work_ms"],
        "quartz_union_ms": q_tot["interval_union_ms"],
        "llama_union_ms": l_tot["interval_union_ms"],
        "unmatched": unmatched_roles(quartz, llama),
        "missing_quartz": missing_roles(quartz),
        "missing_llama": missing_roles(llama),
    }


def overhead_ms(uninstrumented: float, instrumented: float) -> float:
    return float(instrumented) - float(uninstrumented)


def discover_nsight(host_nsys: str | None, host_ncu: str | None) -> dict[str, Any]:
    return {
        "nsys": {
            "available": bool(host_nsys),
            "path": host_nsys,
            "error": None if host_nsys else "nsys not found",
        },
        "ncu": {
            "available": bool(host_ncu),
            "path": host_ncu,
            "error": None if host_ncu else "ncu not found",
        },
        "full_ncu_sweep": False,
        "owner": "OPT-061",
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
