"""OPT-139 NCU counter identity: parse, authenticate, and admit launches.

Diagnostics only. No production kernel or selector changes. Throughput is
never a supported mechanism by itself. Prefill decode-attention counters stay
ineligible until OPT-140.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.performance_evidence import (  # noqa: E402
    classify_kernel_family,
    missing_counter_record,
    parse_nsys_sqlite,
    replay_production_boundary_ok,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt139_counter_identity_contract.json"
ITERATION = ROOT / "pins/opt139_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt139_counter_identity.json"
EVIDENCE = ROOT / "evidence/optimization/opt139-counter-identity"
REPORT = EVIDENCE / "REPORT.md"
REPLAY_BIN = "build/qw38-cuda-component-replay"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
SELECTOR = "decode_segments8"
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
PREFIXES = (128, 2048)
DECODE_SHAPES = (("d128", 128), ("d2048", 2048))
PHASES = ("preflight", "identity", "counters", "report")
CHILD_TIMEOUT_S = 300
PROMPT_FFN_TIMEOUT_S = 1200
AGGREGATE_DEADLINE_S = 7200
CROSSOVER_THRESHOLD = 1024
VERIFIED_MAX = 4096
MMA_THRESHOLD = 8192
PATH_WARP_QUERY = "warp_query"
PATH_VEC128_ONLINE = "vec128_online"
PATH_DENSE_MMA = "dense_bf16_tile_f16_mma_decode_v1"
LAUNCH_WARP_QUERY = "warp_query_decode_attention"
LAUNCH_VEC128_ONLINE = "vec128_online_decode_attention"
LAUNCH_DENSE_MMA = "dense_bf16_tile_f16_mma_decode_v1"
LAUNCH_PREFILL_ATTN = "fattn_mma_pipeline_opt111_base"
LLAMA_KERNEL_STEMS = (
    "flash_attn_stream_k_fixup_general",
    "flash_attn_combine_results",
    "flash_attn_mask_to_KV_max",
    "flash_attn_ext_vec",
    "flash_attn_ext_f16",
    "quantize_mmq_q8_1",
    "mul_mat_q_stream_k_fixup",
    "quantize_q8_1",
    "mul_mat_q",
    "rms_norm_f32",
    "l2_norm_f32",
)
OPT140_INELIGIBLE = "prefill_decode_attention_ineligible_until_opt140"
RESULT_PREFIX = "QW38_OPT139_COUNTER_IDENTITY_RESULT="
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "mode",
    "claims_throughput",
    "claims_performance_improvement",
    "production_kept",
    "selected_execution_graph_path",
    "prefixes",
    "preflight",
    "identity",
    "counters",
    "answers",
    "report_path",
)

SLOT_PREFIXES: dict[str, tuple[str, ...]] = {
    "dram_read_bytes": ("dram__bytes_op_read", "dram__bytes_read"),
    "dram_write_bytes": ("dram__bytes_op_write", "dram__bytes_write"),
    "dram_throughput": ("dram__throughput",),
    "l2_traffic": ("lts__t_sectors",),
    "l2_hit_rate": ("lts__t_sector_hit_rate",),
    "sm_throughput": ("sm__throughput",),
    "tensor_activity": ("sm__pipe_tensor",),
    "achieved_occupancy": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active",
    ),
    "registers": ("launch__registers_per_thread",),
    "local_memory_spills": (
        "launch__local_memory",
        "smsp__sass_lmem_total_bytes",
    ),
}
STALL_PREFIXES = (
    "smsp__warps_issue_stalled_long_scoreboard",
    "smsp__warps_issue_stalled_barrier",
)
SUM_PREFERRED = (
    "dram__bytes",
    "lts__t_sectors",
    "sm__pipe_tensor",
    "launch__local_memory",
    "smsp__sass_lmem",
    "gpu__time_duration",
)
ROUND_LINE = re.compile(
    r"round family=(?P<family>\S+)\s+cache_mode=(?P<cache>\S+)"
    r".*enclosing_ms=(?P<enclosing>[0-9.]+)"
    r".*kernel_only_ms=(?P<kernel>[0-9.]+)"
)
DISPATCH_LINE = re.compile(
    r"decode_attention_dispatch path=(?P<path>\S+) launch=(?P<launch>\S+)"
    r".*decode_position=(?P<position>\d+)"
    r".*path_for_position=(?P<path_for>\S+)"
)
PROF_LAUNCH = re.compile(
    r'==PROF==\s+Profiling\s+"([^"]+)"\s+-\s+(\d+)',
    re.IGNORECASE,
)
TABLE_METRIC = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_.]*)\s+([A-Za-z%/.]+)\s+"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


class CounterIdentityError(RuntimeError):
    """OPT-139 measurement or policy failure."""


@dataclass
class NcuMetric:
    name: str
    value: float | None
    unit: str | None
    raw: str | None = None
    missing_reason: str | None = None


@dataclass
class NcuLaunch:
    launch_id: str | None
    kernel_name: str | None
    metrics: dict[str, NcuMetric] = field(default_factory=dict)
    device: str | None = None
    context: str | None = None
    stream: str | None = None
    grid: str | None = None
    block: str | None = None
    process_name: str | None = None


@dataclass
class NcuParse:
    launches: list[NcuLaunch]
    enclosing_replay: dict[str, Any]
    format: str
    errors: list[str] = field(default_factory=list)


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-139":
        raise CounterIdentityError("counter-identity contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-139", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-139 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def store_sidecar(
    run_dir: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = sidecar(run_dir, name)
    dump_json(path, payload)
    return dict(payload)


def load_sidecar(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-139",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "preflight": None,
        "identity": None,
        "counters": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def current_fixture() -> dict[str, Any]:
    if FIXTURE.is_file():
        payload = load_json(FIXTURE)
        if isinstance(payload, dict) and payload.get("task") == "OPT-139":
            return payload
    return empty_fixture()


def persist_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    dump_json(FIXTURE, payload)
    return dict(payload)


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise CounterIdentityError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise CounterIdentityError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise CounterIdentityError("selector drifted from decode_segments8")


def parse_numeric(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = str(text).strip().strip('"').replace(",", "")
    if cleaned in {"", "n/a", "N/A", "nan", "NaN", "null", "None"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        match = re.match(
            r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
            cleaned,
        )
        if match:
            return float(match.group(1))
        return None


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _header_map(row: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, cell in enumerate(row):
        mapping[_norm_header(cell)] = index
    return mapping


def _cell(row: Sequence[str], headers: Mapping[str, int], *names: str) -> str | None:
    for name in names:
        index = headers.get(_norm_header(name))
        if index is not None and index < len(row):
            value = row[index].strip()
            if value:
                return value
    return None


def _is_csv_header(row: Sequence[str]) -> bool:
    headers = _header_map(row)
    return "metricname" in headers and ("metricvalue" in headers or "value" in headers)


def _csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("==") or stripped.startswith("#"):
            continue
        if stripped.startswith("Device ") and "," not in stripped:
            continue
        try:
            parsed = next(csv.reader(io.StringIO(stripped)))
        except csv.Error:
            continue
        if parsed:
            rows.append(parsed)
    return rows


def _new_launch(
    *,
    launch_id: str | None = None,
    kernel_name: str | None = None,
) -> NcuLaunch:
    return NcuLaunch(launch_id=launch_id, kernel_name=kernel_name)


def _add_metric(
    launch: NcuLaunch, name: str, value: float | None, unit: str | None, raw: str
) -> None:
    if not name:
        return
    launch.metrics[name] = NcuMetric(
        name=name,
        value=value,
        unit=unit,
        raw=raw,
        missing_reason=None if value is not None else "unparseable_metric_value",
    )


def _find_launch(
    launches: Sequence[NcuLaunch],
    *,
    launch_id: str | None,
    kernel_name: str | None,
) -> NcuLaunch | None:
    if launch_id:
        for item in launches:
            if item.launch_id == launch_id:
                return item
    if kernel_name:
        stem = kernel_stem(kernel_name)
        for item in launches:
            if kernel_stem(item.kernel_name) == stem:
                return item
    return None


def parse_ncu_output(text: str) -> NcuParse:
    """Parse NCU CSV or table output into per-launch metric records.

    Launches stay separate. Percentages are never summed. Measured zero stays
    zero; absent metrics stay missing.
    """
    errors: list[str] = []
    launches: list[NcuLaunch] = []
    format_name = "empty"
    csv_headers: dict[str, int] | None = None
    enclosing: dict[str, Any] = {
        "kind": "application_replay_enclosing_totals",
        "separated_from_target_kernel": True,
        "notes": [],
    }
    current = _new_launch()
    table_open = False

    def commit_table() -> None:
        nonlocal current, table_open
        if table_open and (current.metrics or current.kernel_name):
            existing = _find_launch(
                launches,
                launch_id=current.launch_id,
                kernel_name=current.kernel_name,
            )
            if existing is None:
                launches.append(current)
            elif not existing.metrics:
                existing.metrics.update(current.metrics)
                if current.kernel_name and not existing.kernel_name:
                    existing.kernel_name = current.kernel_name
        current = _new_launch()
        table_open = False

    for line in text.splitlines():
        stripped = line.strip()
        if "timeout after" in stripped or stripped.startswith("phase="):
            enclosing["notes"].append(stripped[:240])

    csv_rows = _csv_rows(text)
    saw_csv = False
    for row in csv_rows:
        if _is_csv_header(row):
            csv_headers = _header_map(row)
            saw_csv = True
            format_name = "csv"
            continue
        if csv_headers is None:
            continue
        name = _cell(row, csv_headers, "Metric Name", "MetricName")
        if not name or name.casefold() in {"metric name", "metricname"}:
            continue
        raw_value = _cell(row, csv_headers, "Metric Value", "MetricValue", "Value")
        unit = _cell(row, csv_headers, "Metric Unit", "MetricUnit", "Unit")
        kernel = _cell(row, csv_headers, "Kernel Name", "KernelName", "Kernel")
        launch_id = _cell(row, csv_headers, "ID", "Launch ID", "Invocation")
        target = _find_launch(launches, launch_id=launch_id, kernel_name=kernel)
        if target is None:
            target = _new_launch(launch_id=launch_id, kernel_name=kernel)
            launches.append(target)
        if kernel and not target.kernel_name:
            target.kernel_name = kernel
        if launch_id and not target.launch_id:
            target.launch_id = launch_id
        target.device = _cell(row, csv_headers, "Device") or target.device
        target.context = _cell(row, csv_headers, "Context") or target.context
        target.stream = _cell(row, csv_headers, "Stream") or target.stream
        target.grid = _cell(row, csv_headers, "Grid Size", "Grid") or target.grid
        target.block = _cell(row, csv_headers, "Block Size", "Block") or target.block
        target.process_name = (
            _cell(row, csv_headers, "Process Name", "Process") or target.process_name
        )
        _add_metric(target, name, parse_numeric(raw_value), unit, raw_value or "")

    if not saw_csv:
        current = _new_launch()
        table_open = False
        for line in text.splitlines():
            prof = PROF_LAUNCH.search(line)
            if prof:
                commit_table()
                current = _new_launch(
                    launch_id=prof.group(2),
                    kernel_name=prof.group(1),
                )
                table_open = True
                format_name = "table"
                continue
            match = TABLE_METRIC.match(line)
            if not match:
                continue
            if not table_open:
                current = _new_launch(launch_id="0", kernel_name=None)
                table_open = True
                format_name = "table"
            _add_metric(
                current,
                match.group(1),
                parse_numeric(match.group(3)),
                match.group(2),
                match.group(3),
            )
        commit_table()

    if not launches:
        errors.append("no_ncu_launches_parsed")
    return NcuParse(
        launches=launches,
        enclosing_replay=enclosing,
        format=format_name,
        errors=errors,
    )


def kernel_stem(name: str | None) -> str:
    if not name:
        return ""
    text = name.strip().strip('"')
    for known in (
        LAUNCH_VEC128_ONLINE,
        LAUNCH_WARP_QUERY,
        LAUNCH_DENSE_MMA,
        LAUNCH_PREFILL_ATTN,
        *LLAMA_KERNEL_STEMS,
    ):
        if known in text:
            return known
    if "fattn_mma_pipeline_kernel" in text:
        return LAUNCH_PREFILL_ATTN
    text = re.sub(r"^void\s+", "", text)
    text = re.sub(r"<unnamed>", "", text)
    text = text.split("<", 1)[0]
    text = text.split("(", 1)[0]
    text = text.rsplit("::", 1)[-1]
    return text.strip()


def ncu_kernel_regex(expected_kernel: str | None) -> str:
    """NCU --kernel-name stem. Prefill identity is a Quartz launch name."""
    if expected_kernel == LAUNCH_PREFILL_ATTN:
        return "fattn_mma_pipeline_kernel"
    return kernel_stem(expected_kernel)


def decode_attention_vec128_path_for_position(
    position: int,
    *,
    override: str | None = None,
    crossover_threshold: int = CROSSOVER_THRESHOLD,
    verified_max: int = VERIFIED_MAX,
    mma_enabled: bool = True,
    mma_threshold: int = MMA_THRESHOLD,
) -> str:
    """Mirror cuda/attention_decode_path.cuh for identity checks."""
    if mma_enabled and position >= mma_threshold:
        return PATH_DENSE_MMA
    if override:
        return override
    if (
        crossover_threshold > 0
        and position >= crossover_threshold
        and position <= verified_max
    ):
        return PATH_VEC128_ONLINE
    return PATH_WARP_QUERY


def launch_for_path(path: str) -> str:
    if path == PATH_VEC128_ONLINE:
        return LAUNCH_VEC128_ONLINE
    if path == PATH_DENSE_MMA:
        return LAUNCH_DENSE_MMA
    return LAUNCH_WARP_QUERY


def production_attn_identity(
    prefix: int, *, override: str | None = None
) -> dict[str, Any]:
    path = decode_attention_vec128_path_for_position(prefix, override=override)
    launch = launch_for_path(path)
    return {
        "prefix": prefix,
        "workload": f"D{prefix}",
        "phase": "decode",
        "family": "attn_core",
        "engine": "quartz",
        "replay_family": "decode-attention",
        "path": path,
        "expected_kernel": launch,
        "hybrid_crossover_threshold": CROSSOVER_THRESHOLD,
        "verified_max": VERIFIED_MAX,
        "override": override,
        "selector": SELECTOR,
        "proof_limit": (
            "hybrid dispatch can select vec128_online at D2048; "
            "a warp_query replay is not automatically evidence for that gap"
        ),
    }


def production_prefill_attn_identity() -> dict[str, Any]:
    """P4096 attn_core identity: OPT-111 base launch, prompt-attention replay."""
    return {
        "prefix": 4096,
        "workload": "P4096",
        "phase": "prefill",
        "family": "attn_core",
        "engine": "quartz",
        "replay_family": "prompt-attention",
        "path": "opt111_base",
        "expected_kernel": LAUNCH_PREFILL_ATTN,
        "capacity": 131072,
        "empty_initial_state": True,
        "prefix_reuse": False,
        "final_token_logits_only": True,
        "selector": SELECTOR,
        "complete_family_replay": True,
        "counter_kernel_is_complete_family": False,
        "decode_attention_substitution": False,
        "proof_limit": (
            "complete prompt-attention replay includes split/stage/convert/"
            "core/merge/epilogue; NCU --kernel-name targets "
            "fattn_mma_pipeline_opt111_base only"
        ),
    }


def select_largest_duration_kernel(
    kernels: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, Any] | None:
    """Pick the largest-duration production kernel inside one family."""
    best: dict[str, Any] | None = None
    best_ns = -1
    for row in kernels:
        name = str(row.get("name") or row.get("kernel") or "")
        if classify_kernel_family(name) != family:
            continue
        duration = row.get("duration_ns")
        if duration is None:
            start = row.get("start_ns")
            end = row.get("end_ns")
            if start is not None and end is not None:
                duration = int(end) - int(start)
        if duration is None:
            continue
        duration_ns = int(duration)
        if duration_ns > best_ns:
            best_ns = duration_ns
            best = {
                "name": name,
                "stem": kernel_stem(name),
                "duration_ns": duration_ns,
                "family": family,
            }
    return best


def kernels_from_sqlite(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    tables = parse_nsys_sqlite(path)
    out: list[dict[str, Any]] = []
    for item in tables.kernels:
        out.append(
            {
                "name": item.name,
                "start_ns": item.start_ns,
                "end_ns": item.end_ns,
                "duration_ns": item.duration_ns,
                "family": item.family or classify_kernel_family(item.name),
            }
        )
    return out


def _metric_preference(name: str) -> tuple[int, int, str]:
    lowered = name.casefold()
    sum_rank = 0 if any(lowered.startswith(prefix) for prefix in SUM_PREFERRED) else 1
    suffix_rank = 3
    if lowered.endswith(".sum"):
        suffix_rank = 0 if sum_rank == 0 else 2
    elif lowered.endswith(".avg") or ".avg." in lowered:
        suffix_rank = 0 if sum_rank == 1 else 1
    elif "." not in lowered.split("__")[-1]:
        suffix_rank = 1
    return (sum_rank, suffix_rank, name)


def _pick_metric(launch: NcuLaunch, prefixes: Sequence[str]) -> NcuMetric | None:
    matches: list[NcuMetric] = []
    for name, metric in launch.metrics.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix):
                matches.append(metric)
                break
    if not matches:
        return None
    matches.sort(key=lambda item: _metric_preference(item.name))
    return matches[0]


def typed_slots_from_launch(launch: NcuLaunch | None) -> dict[str, Any]:
    """Populate typed DRAM/L2/SM/warp/stall/register/spill slots.

    Missing or malformed values stay null with reasons. Measured zero stays
    zero. Stall percentages are recorded per metric and never summed.
    """
    slots: dict[str, Any] = {
        "dram_read_bytes": None,
        "dram_write_bytes": None,
        "dram_throughput": None,
        "l2_traffic": None,
        "l2_hit_rate": None,
        "sm_throughput": None,
        "tensor_activity": None,
        "achieved_occupancy": None,
        "stalls": None,
        "registers": None,
        "local_memory_spills": None,
        "units": {},
        "metrics": {},
        "missing_reasons": {},
        "zero_filled": False,
    }
    if launch is None:
        slots["missing_reasons"]["launch"] = "target_launch_absent"
        return slots
    for slot, prefixes in SLOT_PREFIXES.items():
        metric = _pick_metric(launch, prefixes)
        if metric is None:
            slots["missing_reasons"][slot] = "metric_absent_from_capture"
            continue
        slots["metrics"][slot] = {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "raw": metric.raw,
        }
        if metric.value is None:
            slots["missing_reasons"][slot] = metric.missing_reason or "malformed_value"
            continue
        slots[slot] = metric.value
        if metric.unit:
            slots["units"][slot] = metric.unit
    stalls: dict[str, Any] = {}
    for prefix in STALL_PREFIXES:
        metric = _pick_metric(launch, (prefix,))
        if metric is None:
            continue
        key = "long_scoreboard" if "long_scoreboard" in prefix else "barrier"
        stalls[key] = {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "normalized": metric.unit,
            "proof_limit": "do_not_sum_stall_percentages",
        }
    if stalls:
        slots["stalls"] = stalls
    else:
        slots["missing_reasons"]["stalls"] = "metric_absent_from_capture"
    slots["metrics"] = {
        name: {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "raw": metric.raw,
        }
        for name, metric in launch.metrics.items()
    }
    return slots


def _launch_as_dict(launch: NcuLaunch) -> dict[str, Any]:
    payload = asdict(launch)
    payload["stem"] = kernel_stem(launch.kernel_name)
    payload["family"] = (
        classify_kernel_family(launch.kernel_name or "") if launch.kernel_name else None
    )
    return payload


def select_target_launch(
    launches: Sequence[NcuLaunch],
    *,
    expected_kernel: str | None,
    family: str | None = None,
) -> NcuLaunch | None:
    expected_stem = kernel_stem(expected_kernel) if expected_kernel else ""
    if expected_stem:
        matches = [
            item
            for item in launches
            if kernel_stem(item.kernel_name) == expected_stem
            or expected_stem in (item.kernel_name or "")
        ]
        if matches:
            return matches[0]
    if family:
        family_matches = [
            item
            for item in launches
            if classify_kernel_family(item.kernel_name or "") == family
        ]
        if family_matches:
            return family_matches[0]
    return None


def identity_match(
    *,
    phase: str,
    engine: str,
    workload: str,
    kernel: str | None,
    expected_kernel: str | None,
    replay_family: str,
    replay_boundary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mismatches: list[str] = []
    if phase == "prefill" and str(replay_family).startswith("decode-"):
        mismatches.append(OPT140_INELIGIBLE)
    if phase == "decode" and str(replay_family).startswith("prompt-"):
        mismatches.append("decode_prompt_replay_mismatch")
    if engine != "quartz":
        mismatches.append("engine_not_quartz")
    expected_stem = kernel_stem(expected_kernel)
    observed_stem = kernel_stem(kernel)
    if expected_stem and observed_stem and expected_stem != observed_stem:
        mismatches.append("kernel_selector_mismatch")
    if expected_stem and not observed_stem:
        mismatches.append("kernel_identity_unresolved")
    boundary = replay_boundary or replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": phase,
            "replay_family": replay_family,
        }
    )
    if not boundary.get("ok"):
        mismatches.append(str(boundary.get("reason") or "replay_boundary_mismatch"))
    workload_ok = workload.upper() in {"D128", "D2048", "DECODE"} or workload in {
        "d128",
        "d2048",
        "decode",
    }
    if phase == "decode" and not workload_ok:
        mismatches.append("workload_mismatch")
    prefill_ok = workload.upper() in {"P4096", "PREFILL"} or workload in {
        "p4096",
        "prefill",
        "prompt-attention",
    }
    if phase == "prefill" and replay_family == "prompt-attention" and not prefill_ok:
        mismatches.append("workload_mismatch")
    ok = not mismatches
    return {
        "ok": ok,
        "phase": phase,
        "engine": engine,
        "workload": workload,
        "kernel": kernel,
        "expected_kernel": expected_kernel,
        "replay_family": replay_family,
        "replay_boundary": boundary,
        "mismatches": mismatches,
        "reason": None if ok else ",".join(mismatches),
    }


def admit_supported_mechanism(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit a mechanism only with identity + source/SASS + counters.

    A throughput value alone cannot establish a mechanism. Prefill
    decode-attention remains ineligible until OPT-140.
    """
    phase = str(record.get("phase") or "")
    replay_family = str(record.get("replay_family") or "")
    if phase == "prefill" and replay_family.startswith("decode-"):
        return {
            "ok": False,
            "supported_mechanism": None,
            "candidate": None,
            "reason": OPT140_INELIGIBLE,
        }
    identity = record.get("identity") or identity_match(
        phase=phase,
        engine=str(record.get("engine") or ""),
        workload=str(record.get("workload") or record.get("shape") or ""),
        kernel=record.get("kernel") or record.get("target_kernel"),
        expected_kernel=record.get("expected_kernel"),
        replay_family=replay_family,
        replay_boundary=record.get("replay_boundary") or record.get("boundary"),
    )
    if not identity.get("ok"):
        return {
            "ok": False,
            "supported_mechanism": None,
            "candidate": None,
            "reason": identity.get("reason") or "identity_rejected",
            "identity": identity,
        }
    throughput_present = any(
        record.get(key) not in (None, False, 0, 0.0)
        for key in ("dram_throughput", "sm_throughput")
    )
    source = record.get("source_observations") or record.get("named_source")
    sass = record.get("sass_observations") or record.get("sass")
    counters_ok = any(
        record.get(key) is not None
        for key in (
            "dram_read_bytes",
            "dram_write_bytes",
            "l2_traffic",
            "tensor_activity",
            "achieved_occupancy",
            "stalls",
            "registers",
            "local_memory_spills",
        )
    )
    if throughput_present and not (source and sass and counters_ok):
        return {
            "ok": False,
            "supported_mechanism": None,
            "candidate": None,
            "reason": "throughput_alone_cannot_establish_mechanism",
            "identity": identity,
        }
    if not (source and sass and counters_ok):
        return {
            "ok": False,
            "supported_mechanism": None,
            "candidate": None,
            "reason": "named_source_sass_and_corroborating_counters_required",
            "identity": identity,
        }
    return {
        "ok": True,
        "supported_mechanism": record.get("proposed_mechanism")
        or "hardware_counter_bound",
        "candidate": record.get("candidate"),
        "reason": None,
        "identity": identity,
    }


def counter_record_from_ncu_blob(
    blob: str,
    *,
    kernel_id: str,
    engine: str,
    family: str,
    phase: str,
    replay_family: str,
    expected_kernel: str | None = None,
    workload: str | None = None,
    prefix: int | None = None,
    selected_metrics: Sequence[str] | None = None,
    command: Sequence[str] | None = None,
    timeout_s: int | None = None,
    raw_artifact: str | None = None,
    binary_hash: str | None = None,
) -> dict[str, Any]:
    """Turn a full NCU blob into typed per-launch counters.

    Enclosing replay totals stay on a sibling object; they are never mixed
    into the target-kernel slots.
    """
    parsed = parse_ncu_output(blob)
    target = select_target_launch(
        parsed.launches,
        expected_kernel=expected_kernel,
        family=family,
    )
    slots = typed_slots_from_launch(target)
    identity = identity_match(
        phase=phase,
        engine=engine,
        workload=workload or (f"D{prefix}" if prefix is not None else phase),
        kernel=target.kernel_name if target else expected_kernel,
        expected_kernel=expected_kernel,
        replay_family=replay_family,
        replay_boundary=None,
    )
    admission = admit_supported_mechanism(
        {
            "phase": phase,
            "engine": engine,
            "workload": workload or (f"D{prefix}" if prefix is not None else phase),
            "replay_family": replay_family,
            "kernel": target.kernel_name if target else None,
            "expected_kernel": expected_kernel,
            "identity": identity,
            **{key: slots.get(key) for key in SLOT_PREFIXES},
            "stalls": slots.get("stalls"),
        }
    )
    other_launches = [
        _launch_as_dict(item) for item in parsed.launches if item is not target
    ]
    return {
        "kernel": kernel_id,
        "engine": engine,
        "family": family,
        "phase": phase,
        "workload": workload or (f"D{prefix}" if prefix is not None else None),
        "prefix": prefix,
        "visible_position": prefix,
        "replay_family": replay_family,
        "replay_mode": "application",
        "expected_kernel": expected_kernel,
        "target_kernel": target.kernel_name if target else None,
        "target_launch_id": target.launch_id if target else None,
        "launch_identity": None if target is None else _launch_as_dict(target),
        "other_launches": other_launches,
        "enclosing_replay": parsed.enclosing_replay,
        "parse_format": parsed.format,
        "parse_errors": parsed.errors,
        "selected_metrics": list(selected_metrics or []),
        "timeout_s": timeout_s,
        "command": list(command or []),
        "raw_artifact": raw_artifact,
        "binary_hash": binary_hash,
        "selector": SELECTOR,
        "metrics": slots.get("metrics") or {},
        "dram_read_bytes": slots.get("dram_read_bytes"),
        "dram_write_bytes": slots.get("dram_write_bytes"),
        "dram_throughput": slots.get("dram_throughput"),
        "l2_traffic": slots.get("l2_traffic"),
        "l2_hit_rate": slots.get("l2_hit_rate"),
        "sm_throughput": slots.get("sm_throughput"),
        "tensor_activity": slots.get("tensor_activity"),
        "achieved_occupancy": slots.get("achieved_occupancy"),
        "stalls": slots.get("stalls"),
        "registers": slots.get("registers"),
        "local_memory_spills": slots.get("local_memory_spills"),
        "units": slots.get("units") or {},
        "missing_reasons": slots.get("missing_reasons") or {},
        "error": None if target is not None else "target_kernel_launch_absent",
        "zero_filled": False,
        "full_ncu_sweep": False,
        "identity": identity,
        "admission": admission,
        "supported_mechanism": admission.get("supported_mechanism"),
        "candidate": admission.get("candidate"),
        "stdout_tail": blob[-1500:],
        "proof_limit": (
            "typed slots come from the identified target launch only; "
            "throughput alone is not a mechanism; occupancy-only is not "
            "a bandwidth claim"
        ),
    }


def parse_enclosing_replay(text: str) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = ROUND_LINE.search(line)
        if not match:
            continue
        rounds.append(
            {
                "family": match.group("family"),
                "cache_mode": match.group("cache"),
                "enclosing_ms": parse_numeric(match.group("enclosing")),
                "kernel_only_ms": parse_numeric(match.group("kernel")),
                "warmup": "warmup=true" in line,
            }
        )
    measured = [row for row in rounds if not row.get("warmup")]
    return {
        "kind": "application_replay_enclosing_totals",
        "separated_from_target_kernel": True,
        "rounds": rounds,
        "measured_enclosing_ms": [row["enclosing_ms"] for row in measured],
        "measured_kernel_only_ms": [row["kernel_only_ms"] for row in measured],
        "notes": ["enclosing replay totals are not target-kernel counters"],
    }


def parse_replay_dispatch(text: str) -> dict[str, Any] | None:
    match = None
    for line in text.splitlines():
        found = DISPATCH_LINE.search(line)
        if found:
            match = found
    if match is None:
        return None
    return {
        "path": match.group("path"),
        "launch": match.group("launch"),
        "decode_position": int(match.group("position")),
        "path_for_position": match.group("path_for"),
    }


def export_ncu_csv(report_path: Path) -> str:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138

    rel = report_path if not report_path.is_absolute() else Path(relpath(report_path))
    command = [
        *opt138._docker_ncu(),
        "ncu",
        "--import",
        str(rel),
        "--csv",
    ]
    completed = opt136.with_gpu_lock(command, timeout_s=CHILD_TIMEOUT_S)
    return (completed.stdout or "") + (completed.stderr or "")


def capture_reuse_valid(
    path: Path, *, expected_kernel: str, metrics: Sequence[str]
) -> bool:
    report = path
    if path.suffix == ".txt":
        report = path.with_suffix(".ncu-rep")
        if report.name.endswith(".ncu.ncu-rep"):
            report = path.parent / path.name.replace(".ncu.txt", ".ncu-rep")
    csv_path = path.parent / path.name.replace(".ncu.txt", ".ncu.csv")
    if csv_path.is_file():
        parsed = parse_ncu_output(
            csv_path.read_text(encoding="utf-8", errors="replace")
        )
        target = select_target_launch(parsed.launches, expected_kernel=expected_kernel)
        return target is not None and bool(target.metrics)
    if report.is_file():
        return True
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if "ERR_NVGPUCTRPERM" in text or "timeout after" in text:
        return False
    parsed = parse_ncu_output(text)
    target = select_target_launch(parsed.launches, expected_kernel=expected_kernel)
    if target is None:
        return False
    if parsed.format not in {"csv", "table", "prof_sections"}:
        return False
    if metrics and not target.metrics:
        return False
    return True


def ncu_collect_command(
    *,
    metrics: Sequence[str],
    kernel_regex: str,
    decode_position: int,
    raw_report: Path | None = None,
) -> list[str]:
    from tools import opt138_remaining_gap_profile as opt138

    command = [
        *opt138._docker_ncu(),
        "ncu",
        "--csv",
        "--metrics",
        ",".join(metrics),
        "--target-processes",
        "all",
        "--replay-mode",
        "application",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--launch-count",
        "1",
    ]
    if raw_report is not None:
        output = raw_report
        if output.is_absolute():
            output = Path(relpath(output))
        command.extend(["-o", str(output), "--force-overwrite"])
    command.extend(
        [
            f"./{REPLAY_BIN}",
            MODEL,
            "--workload",
            "decode-attention",
            "--cache-mode",
            "rotating",
            "--opt138-protocol",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--decode-position",
            str(decode_position),
            "--evidence-dir",
            "evidence/optimization/opt139-counter-identity/replay",
        ]
    )
    return command


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-139",
        "phase": phase,
        "mode": mode,
        "ok": False,
        "status": "incomplete",
        "blocked": True,
        "reason": preflight.get("blocked") or ["gpu_or_ncu_unavailable"],
        "gpu_available": preflight.get("gpu_available"),
        "ncu_available": (preflight.get("ncu") or {}).get("ncu_available"),
        "claims_throughput": False,
        "family_plan": family_plan(mode, phase if phase in PHASES else "report"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, f"{phase}.json", payload)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138
    from tools.opt080_batch_gate import IMAGE, docker_common

    run_dir.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    pins = opt138.authenticate_opt138_pins()
    make_rc = 1
    make_stderr = gpu_blocker or ""
    if available:
        import subprocess

        make = subprocess.run(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt139-diagnostics"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stderr = (make.stderr or "")[-2000:]
    ncu = {"ncu_available": False, "error": "gpu_unavailable"}
    if available:
        ncu = opt138._query_ncu_metrics()
    replay_hash = opt136.sha256_file(ROOT / REPLAY_BIN)
    blocked: list[str] = []
    if not available:
        blocked.append(gpu_blocker or "gpu_unavailable")
    if not ncu.get("ncu_available"):
        blocked.append(str(ncu.get("error") or "ncu_unavailable"))
    if not (ROOT / REPLAY_BIN).is_file():
        blocked.append("replay_bin_missing")
    if not pins.get("ok"):
        blocked.append("stale_parent_identity")
    ok = (
        available
        and bool(ncu.get("ncu_available"))
        and (ROOT / REPLAY_BIN).is_file()
        and bool(pins.get("ok"))
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-139",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "ncu": ncu,
        "hashes": {"replay": replay_hash},
        "make_returncode": make_rc,
        "make_stderr_tail": make_stderr,
        "authenticated_current_pins": pins,
        "parent": PARENT,
        "replay_bin": REPLAY_BIN,
        "child_timeout_s": CHILD_TIMEOUT_S,
        "prompt_ffn_timeout_s": PROMPT_FFN_TIMEOUT_S,
        "aggregate_deadline_s": AGGREGATE_DEADLINE_S,
        "claims_throughput": False,
        "blocked": blocked,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def _sqlite_candidates(prefix: int) -> list[Path]:
    traces = ROOT / "build/optimization-runs/opt138/traces"
    names = [
        traces / f"quartz-d{prefix}-node-r0.2.sqlite",
        traces / f"quartz-d{prefix}-node-r0.1.sqlite",
        traces / f"quartz-d{prefix}-node-r1.2.sqlite",
    ]
    fixture = current_opt138_sqlite(prefix)
    if fixture:
        names.insert(0, fixture)
    return names


def current_opt138_sqlite(prefix: int) -> Path | None:
    fixture_path = ROOT / "fixtures/opt138_remaining_gap_profile.json"
    if not fixture_path.is_file():
        return None
    payload = load_json(fixture_path)
    captures = payload.get("captures") or {}
    key = f"quartz-d{prefix}-node-r0"
    row = captures.get(key) or {}
    exported = row.get("sqlite_by_window") or {}
    middle = exported.get("middle")
    if middle:
        path = ROOT / str(middle)
        if path.is_file():
            return path
    return None


def run_identity(run_dir: Path, mode: str) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for key, prefix in DECODE_SHAPES:
        expected = production_attn_identity(prefix)
        sqlite_path = next(
            (path for path in _sqlite_candidates(prefix) if path.is_file()), None
        )
        largest = None
        kernel_source = "production_selector"
        if sqlite_path is not None:
            largest = select_largest_duration_kernel(
                kernels_from_sqlite(sqlite_path),
                "attn_core",
            )
            if largest:
                kernel_source = "trace_largest_duration"
                expected["trace_kernel"] = largest
                if kernel_stem(
                    str(largest.get("stem") or largest.get("name"))
                ) and kernel_stem(
                    str(largest.get("stem") or largest.get("name"))
                ) != kernel_stem(expected["expected_kernel"]):
                    expected["trace_selector_tension"] = (
                        "largest-duration family kernel differs from hybrid path launch"
                    )
        identity = identity_match(
            phase="decode",
            engine="quartz",
            workload=f"D{prefix}",
            kernel=expected["expected_kernel"],
            expected_kernel=expected["expected_kernel"],
            replay_family="decode-attention",
            replay_boundary=None,
        )
        warp_as_d2048 = None
        if prefix == 2048:
            warp_as_d2048 = identity_match(
                phase="decode",
                engine="quartz",
                workload="D2048",
                kernel=LAUNCH_WARP_QUERY,
                expected_kernel=expected["expected_kernel"],
                replay_family="decode-attention",
                replay_boundary=None,
            )
        shapes[key] = {
            **expected,
            "kernel_source": kernel_source,
            "sqlite": None if sqlite_path is None else relpath(sqlite_path),
            "identity": identity,
            "warp_query_is_not_d2048_evidence": warp_as_d2048,
        }
    prefill = {
        "phase": "prefill",
        "family": "attn_core",
        "eligible": False,
        "reason": OPT140_INELIGIBLE,
        "replay_family": "decode-attention",
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-139",
        "phase": "identity",
        "mode": mode,
        "ok": True,
        "decode_shapes": shapes,
        "prefill_attn_core": prefill,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "identity"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "identity.json", payload)
    return store_sidecar(run_dir, "identity.json", payload)


def run_counters(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138

    preflight = load_sidecar(run_dir, "preflight.json") or {}
    identity = load_sidecar(run_dir, "identity.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("counters", run_dir, mode, preflight)
    ncu = preflight.get("ncu") or opt138._query_ncu_metrics()
    metrics = list(ncu.get("selected_metrics") or [])
    raw_dir = EVIDENCE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for key, prefix in DECODE_SHAPES:
        shape = (identity.get("decode_shapes") or {}).get(key) or {}
        expected_kernel = str(
            shape.get("expected_kernel")
            or launch_for_path(decode_attention_vec128_path_for_position(prefix))
        )
        kernel_id = f"quartz:attn_core:D{prefix}"
        if not ncu.get("ncu_available"):
            records.append(
                {
                    **missing_counter_record(
                        kernel=kernel_id,
                        engine="quartz",
                        error=str(ncu.get("error") or "ncu_unavailable"),
                    ),
                    "phase": "decode",
                    "family": "attn_core",
                    "prefix": prefix,
                    "workload": f"D{prefix}",
                    "expected_kernel": expected_kernel,
                    "supported_mechanism": None,
                    "candidate": None,
                }
            )
            continue
        raw_txt = raw_dir / f"d{prefix}-attn_core.ncu.txt"
        raw_csv = raw_dir / f"d{prefix}-attn_core.ncu.csv"
        report_path = raw_dir / f"d{prefix}-attn_core"
        report_file = raw_dir / f"d{prefix}-attn_core.ncu-rep"
        reused = False
        blob = ""
        command: list[str] = []
        timeout_s = CHILD_TIMEOUT_S
        if capture_reuse_valid(
            raw_txt, expected_kernel=expected_kernel, metrics=metrics
        ):
            reused = True
            command = [
                "reused",
                str(report_file if report_file.is_file() else raw_txt),
            ]
            if raw_txt.is_file():
                blob = raw_txt.read_text(encoding="utf-8", errors="replace")
        else:
            command = ncu_collect_command(
                metrics=metrics,
                kernel_regex=kernel_stem(expected_kernel),
                decode_position=prefix,
                raw_report=report_path,
            )
            completed = opt136.with_gpu_lock(command, timeout_s=timeout_s)
            blob = (completed.stdout or "") + (completed.stderr or "")
            raw_txt.write_text(blob, encoding="utf-8")
            if (
                completed.returncode != 0
                or "ERR_NVGPUCTRPERM" in blob
                or "timeout after" in blob
            ) and not report_file.is_file():
                records.append(
                    {
                        **missing_counter_record(
                            kernel=kernel_id,
                            engine="quartz",
                            error=blob[-800:] or f"ncu_rc={completed.returncode}",
                        ),
                        "phase": "decode",
                        "family": "attn_core",
                        "prefix": prefix,
                        "workload": f"D{prefix}",
                        "replay_family": "decode-attention",
                        "expected_kernel": expected_kernel,
                        "command": command,
                        "timeout_s": timeout_s,
                        "raw_artifact": relpath(raw_txt),
                        "reused": reused,
                        "supported_mechanism": None,
                        "candidate": None,
                    }
                )
                continue
        if report_file.is_file() and not (
            raw_csv.is_file() and raw_csv.stat().st_size > 0
        ):
            raw_csv.write_text(export_ncu_csv(report_file), encoding="utf-8")
        parse_text = (
            raw_csv.read_text(encoding="utf-8", errors="replace")
            if raw_csv.is_file()
            else blob
        )
        record = counter_record_from_ncu_blob(
            parse_text,
            kernel_id=kernel_id,
            engine="quartz",
            family="attn_core",
            phase="decode",
            replay_family="decode-attention",
            expected_kernel=expected_kernel,
            workload=f"D{prefix}",
            prefix=prefix,
            selected_metrics=metrics,
            command=command,
            timeout_s=timeout_s,
            raw_artifact=relpath(raw_csv if raw_csv.is_file() else raw_txt),
            binary_hash=(preflight.get("hashes") or {}).get("replay"),
        )
        if blob:
            record["enclosing_replay"] = parse_enclosing_replay(blob)
            dispatch = parse_replay_dispatch(blob)
            if dispatch:
                record["replay_dispatch"] = dispatch
        record["ncu_report"] = relpath(report_file) if report_file.is_file() else None
        record["reused"] = reused
        records.append(record)
    prefill = {
        "kernel": "quartz:attn_core:P4096",
        "engine": "quartz",
        "family": "attn_core",
        "phase": "prefill",
        "eligible": False,
        "error": OPT140_INELIGIBLE,
        "supported_mechanism": None,
        "candidate": None,
        "zero_filled": False,
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-139",
        "phase": "counters",
        "mode": mode,
        "ok": True,
        "ncu": ncu,
        "kernels": records,
        "prefill_attn_core": prefill,
        "full_ncu_sweep": False,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "counters"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "counter-evidence.json", payload)
    return store_sidecar(run_dir, "counters.json", payload)


def write_report(payload: Mapping[str, Any]) -> None:
    preflight = payload.get("preflight") or {}
    identity = payload.get("identity") or {}
    counters = payload.get("counters") or {}
    lines = [
        "# OPT-139 — Normalize NCU evidence and authenticate production kernel identity",
        "",
        "Diagnostics only. `claims_throughput=false`. Shipping delta: N/A. NLL: N/A.",
        "No production kernel or selector change. No throughput keep claim.",
        "",
        f"Measured at `{payload.get('measurement_utc')}`. Parent `{PARENT}`.",
        f"GPU available `{preflight.get('gpu_available')}`. "
        f"NCU `{(preflight.get('ncu') or {}).get('ncu_available')}`.",
        "",
        "## Identity",
        "",
    ]
    shapes = identity.get("decode_shapes") or {}
    for key, prefix in DECODE_SHAPES:
        row = shapes.get(key) or {}
        ident = row.get("identity") or {}
        lines.append(
            f"- D{prefix} path=`{row.get('path')}` expected_kernel="
            f"`{row.get('expected_kernel')}` source=`{row.get('kernel_source')}` "
            f"identity_ok=`{ident.get('ok')}`"
        )
        if prefix == 2048:
            warp = row.get("warp_query_is_not_d2048_evidence") or {}
            lines.append(
                f"  warp_query as D2048 evidence rejected: `{warp.get('reason')}`"
            )
    prefill = identity.get("prefill_attn_core") or {}
    lines.extend(
        [
            f"- P4096 attn_core eligible=`{prefill.get('eligible')}` "
            f"reason=`{prefill.get('reason')}`",
            "",
            "## Typed counters",
            "",
        ]
    )
    for row in counters.get("kernels") or []:
        kernel = kernel_stem(str(row.get("target_kernel") or "")) or row.get(
            "target_kernel"
        )
        units = row.get("units") or {}
        lines.append(
            f"- {row.get('workload')} kernel=`{kernel}` "
            f"raw=`{row.get('raw_artifact')}` report=`{row.get('ncu_report')}` "
            f"reused=`{row.get('reused')}` "
            f"dram_read=`{row.get('dram_read_bytes')}` "
            f"{units.get('dram_read_bytes') or ''} "
            f"dram_write=`{row.get('dram_write_bytes')}` "
            f"dram_throughput=`{row.get('dram_throughput')}`% "
            f"l2=`{row.get('l2_traffic')}` {units.get('l2_traffic') or ''} "
            f"sm=`{row.get('sm_throughput')}`% "
            f"warps=`{row.get('achieved_occupancy')}`% "
            f"tensor=`{row.get('tensor_activity')}` "
            f"registers=`{row.get('registers')}` spills=`{row.get('local_memory_spills')}` "
            f"mechanism=`{row.get('supported_mechanism')}` "
            f"admission=`{(row.get('admission') or {}).get('reason')}`"
        )
    lines.extend(
        [
            "",
            f"Prefill attn_core: `{(counters.get('prefill_attn_core') or {}).get('error')}`.",
            "",
            "## Status",
            "",
            f"status=`{payload.get('status')}` blocked=`{payload.get('gpu_phases_blocked')}`.",
            "Unsupported or identity-rejected mechanisms remain explicitly unknown.",
            "Throughput alone cannot establish a mechanism. Candidate stays null.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    identity = load_sidecar(run_dir, "identity.json") or {}
    counters = load_sidecar(run_dir, "counters.json") or {}
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    parsed_ok = any(
        row.get("target_kernel") and row.get("error") in (None, "")
        for row in (counters.get("kernels") or [])
    )
    identity_ok = (
        all(
            ((row.get("identity") or {}).get("ok") is True)
            for row in (identity.get("decode_shapes") or {}).values()
        )
        if identity.get("decode_shapes")
        else False
    )
    status = "structure"
    if blocked:
        status = "incomplete"
    elif parsed_ok and identity_ok:
        status = "measured"
    elif identity_ok:
        status = "incomplete"
    answers = {
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "supported_mechanism": None,
        "candidate": None,
        "decode": {
            key: {
                "prefix": prefix,
                "path": ((identity.get("decode_shapes") or {}).get(key) or {}).get(
                    "path"
                ),
                "expected_kernel": (
                    (identity.get("decode_shapes") or {}).get(key) or {}
                ).get("expected_kernel"),
                "supported_mechanism": None,
                "candidate": None,
            }
            for key, prefix in DECODE_SHAPES
        },
        "prefill_attn_core": OPT140_INELIGIBLE,
    }
    if not (EVIDENCE / "counter-evidence.json").is_file():
        dump_json(
            EVIDENCE / "counter-evidence.json",
            counters
            or {
                "schema_version": 1,
                "task": "OPT-139",
                "status": "incomplete",
                "kernels": [],
            },
        )
    if not (EVIDENCE / "identity.json").is_file():
        dump_json(EVIDENCE / "identity.json", identity or {})
    payload = {
        "schema_version": 1,
        "task": "OPT-139",
        "status": status,
        "mode": mode,
        "measurement_utc": utc_now(),
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefixes": list(PREFIXES),
        "preflight": preflight,
        "identity": identity,
        "counters": counters,
        "answers": answers,
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
        "family_plan": family_plan(mode, "report"),
        "gpu_phases_blocked": blocked,
    }
    validate_fixture(payload)
    persist_fixture(payload)
    write_report(payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "identity":
        result = run_identity(run_dir, mode)
    elif phase == "counters":
        result = run_counters(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise CounterIdentityError(f"unknown phase {phase}")
    elapsed = time.time() - started
    result = dict(result)
    result["phase_elapsed_s"] = elapsed
    if elapsed > AGGREGATE_DEADLINE_S:
        result["aggregate_deadline_exceeded"] = True
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    result = run_phase(args.phase, Path(args.run_dir), args.mode)
    json.dump(
        {
            "phase": args.phase,
            "ok": bool(result.get("ok", True)),
            **({"status": result.get("status")} if result.get("status") else {}),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.stdout.write(
        RESULT_PREFIX
        + json.dumps({"phase": args.phase, "ok": bool(result.get("ok", True))})
        + "\n"
    )
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CounterIdentityError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
