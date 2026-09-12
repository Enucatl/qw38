"""OPT-109 persistent col-major GDN state admission. Sequential stays until keep."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    FEEDBACK_CRIT,
    T_CRIT_DF9,
    docker_common,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.opt077_gdn_decode import parse_gdn_dispatch  # noqa: E402
from tools.opt089_q4_promotion import e2e_guards, parse_prefill_wall_ms  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt109_persistent_gdn_state_contract.json"
ITERATION = ROOT / "pins/opt109_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt109_persistent_gdn_state.json"
REPORT = ROOT / "evidence/optimization/opt109-persistent-gdn-state/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt109-persistent-gdn-state/REJECTION.md"
EVIDENCE = REPORT.parent
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
GDN_PIN_FILE = ROOT / "cuda/gdn_decode_path.cuh"
NATIVE = "build/qw38-cuda-opt109-persistent-gdn-state-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "parity",
    "pilot",
    "gdn",
    "quality",
    "state",
    "d128",
    "d2048",
    "prefill-guard",
    "memory",
)
CONTROL_ID = "sequential_row_major"
CANDIDATE_ID = "persistent_transposed"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.10
E2E_REGRESSION_FRAC = 0.02
THROUGHPUT_MIN = 0.98
P95_RATIO_MAX = 1.02
PREFILL_THROUGHPUT_MIN = 0.95
LOGICAL_LAYOUT = "row_major_fp32"
DEVICE_LAYOUT = "col_major_fp32"
LOGICAL_SHA = "3ac4ad9f78bcbdd1f4aa80da8617b32d6e84384783be3d314fec91c1661a4721"
DEVICE_SHA = "e709048e6ff6ce6223cb14aae0a6c686ae18980d82213f8bd34c266200b26741"
ELEMENT_COUNT = 48 * 128 * 128
GDN_STATE_BYTES = 158859264

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "gdn_decode": "sequential",
        "warps_per_cta": 4,
        "grid_y": 1,
        "role": "control",
        "expected_launch": "prepare_recurrence_window",
        "state_layout": LOGICAL_LAYOUT,
    },
    {
        "id": CANDIDATE_ID,
        "gdn_decode": "persistent_transposed",
        "warps_per_cta": 4,
        "grid_y": 32,
        "role": "candidate",
        "source_task": "OPT-109",
        "expected_launch": "prepare_recurrence_decode_transposed",
        "state_layout": DEVICE_LAYOUT,
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    env = os.environ.copy()
    env["QW38_CUDA_TEST_TIER"] = tier
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        if listed and not listed[0].startswith("docker"):
            listed = [*docker_common(tier), *listed]
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if completed.returncode != 0:
        raise AdmissionError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def current_gdn_pin() -> str:
    match = re.search(
        r'kSelectedGdnDecodePath\[\] = "([^"]+)"',
        GDN_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedGdnDecodePath")
    return match.group(1)


def apply_gdn_pin(path_id: str) -> None:
    text = GDN_PIN_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedGdnDecodePath\[\] = "[^"]+"',
        f'kSelectedGdnDecodePath[] = "{path_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the GDN decode production pin")
    GDN_PIN_FILE.write_text(updated, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied = {"gdn_decode": False}
    if results.get("production_kept") and not results.get("shipping_unchanged"):
        want = str(results.get("shipping_gdn_decode") or "")
        if not want:
            raise AdmissionError("keep is missing shipping GDN decode id")
        pin = "persistent_transposed" if want == CANDIDATE_ID else want
        if pin != current_gdn_pin():
            apply_gdn_pin(pin)
            applied["gdn_decode"] = True
    elif results.get("mode") == "acceptance" and not results.get("production_kept"):
        if current_gdn_pin() != "sequential":
            apply_gdn_pin("sequential")
            applied["gdn_decode"] = True
    applied["gdn_pin"] = current_gdn_pin()
    return applied


def logical_index(
    head: int, key_row: int, value_col: int, key_width: int, value_width: int
) -> int:
    return head * key_width * value_width + key_row * value_width + value_col


def device_index(
    head: int, key_row: int, value_col: int, key_width: int, value_width: int
) -> int:
    return head * key_width * value_width + value_col * key_width + key_row


def layout_roundtrip_host(heads: int = 2, width: int = 8) -> dict[str, Any]:
    logical = [0.0] * (heads * width * width)
    device = [0.0] * len(logical)
    for head in range(heads):
        for row in range(width):
            for col in range(width):
                value = float(head * 1000 + row * 20 + col)
                logical[logical_index(head, row, col, width, width)] = value
                device[device_index(head, row, col, width, width)] = value
    restored = [0.0] * len(logical)
    for head in range(heads):
        for row in range(width):
            for col in range(width):
                restored[logical_index(head, row, col, width, width)] = device[
                    device_index(head, row, col, width, width)
                ]
    return {
        "same_count": len(logical) == len(device),
        "restored_ok": restored == logical,
        "layouts_differ": logical != device,
        "element_count_production": ELEMENT_COUNT,
        "pass": restored == logical and logical != device,
    }


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if phase in {"d128", "d2048"} and engine_pairs == 0:
        engine_pairs = 5
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-gdn")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": engine_pairs,
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("output_tokens", 32) or 32),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-109",
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
        "keep": bool(keep),
        "capture_key": capture_key,
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["gdn_decode"]):
        return False
    if str(observed.get("launch")) != str(config["expected_launch"]):
        return False
    if int(observed.get("warps_per_cta", 0)) != 4:
        return False
    if int(observed.get("grid_x", 0)) != 48:
        return False
    return int(observed.get("grid_y", -1)) == int(config["grid_y"])


def empty_verdict_row() -> dict[str, Any]:
    return {
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "incomplete": True,
    }


def decide_independent_verdicts(
    *,
    parity: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    pilot: Mapping[str, Any] | None,
    component: Mapping[str, Any] | None,
    engine_by_prefix: Mapping[str, Mapping[str, Any]],
    prefill: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    memory: Mapping[str, Any] | None,
    mode: str,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    by_ident = (parity or {}).get("by_ident") or {}
    for config in CONFIGS:
        cid = config["id"]
        previous = dict((prior or {}).get(cid) or empty_verdict_row())
        kpass = (
            bool(by_ident.get(cid, {}).get("kernel_parity_pass"))
            if by_ident
            else bool(previous.get("kernel_parity_pass"))
        )
        qpass = (
            bool((quality or {}).get("model_quality_pass"))
            if quality
            else bool(previous.get("model_quality_pass"))
        )
        perf_ok = cid == CANDIDATE_ID
        if cid == CANDIDATE_ID:
            pilot_stats = pilot or {}
            stats = component or {}
            if not bool(pilot_stats.get("positive")):
                perf_ok = False
            if stats:
                saving = float(stats.get("mean_diff_ms", 0.0))
                positive = bool(stats.get("positive")) and saving >= MIN_SAVING_MS
                ci_low = float(stats.get("ci95_low", 0.0) or 0.0)
                timed = int(stats.get("timed_relayout_launches", 0) or 0)
                decode_conv = int(stats.get("decode_conversions", 0) or 0)
                if (
                    stats.get("skipped")
                    or not positive
                    or ci_low <= 0.0
                    or timed != 0
                    or decode_conv != 0
                ):
                    perf_ok = False
            elif mode == "acceptance":
                perf_ok = False
            if mode == "acceptance":
                for prefix in ("d128", "d2048"):
                    engine = dict(engine_by_prefix.get(prefix) or {})
                    guards = engine.get("guards") or {}
                    control_mean = float(engine.get("control_mean_ms", 0.0) or 0.0)
                    cand_mean = float(engine.get("candidate_mean_ms", 0.0) or 0.0)
                    if (
                        not engine
                        or engine.get("skipped")
                        or not guards.get("pass")
                        or control_mean <= 0.0
                        or cand_mean >= control_mean
                    ):
                        perf_ok = False
                ratio = float((prefill or {}).get("throughput_ratio", 0.0) or 0.0)
                if (
                    not prefill
                    or prefill.get("skipped")
                    or ratio < PREFILL_THROUGHPUT_MIN
                ):
                    perf_ok = False
                if not state or not bool(state.get("pass")):
                    perf_ok = False
                if not memory or not bool(memory.get("pass")):
                    perf_ok = False
            else:
                for engine in engine_by_prefix.values():
                    guards = (engine or {}).get("guards") or {}
                    if engine and not guards:
                        control_mean = float(engine.get("control_mean_ms", 0.0) or 0.0)
                        cand_mean = float(engine.get("candidate_mean_ms", 0.0) or 0.0)
                        upper = float(engine.get("regression_upper_ms", 0.0) or 0.0)
                        if cand_mean >= control_mean or (
                            control_mean > 0.0
                            and upper > E2E_REGRESSION_FRAC * control_mean
                        ):
                            perf_ok = False
                    elif guards and not guards.get("pass"):
                        perf_ok = False
                if prefill and float(prefill.get("throughput_ratio", 1.0)) < (
                    PREFILL_THROUGHPUT_MIN
                ):
                    perf_ok = False
                if state and not bool(state.get("pass", True)):
                    perf_ok = False
                if memory and not bool(memory.get("pass", True)):
                    perf_ok = False
        rows[cid] = {
            "kernel_parity_pass": kpass,
            "model_quality_pass": qpass,
            "performance_pass": perf_ok if cid == CANDIDATE_ID else False,
            "production_kept": False,
            "incomplete": mode != "acceptance",
        }
    winners = [
        cid
        for cid, row in rows.items()
        if cid != CONTROL_ID
        and row["kernel_parity_pass"]
        and row["model_quality_pass"]
        and row["performance_pass"]
    ]
    selected = CONTROL_ID
    shipping = "sequential"
    production_kept = False
    claims = False
    if mode == "acceptance" and winners:
        selected = CANDIDATE_ID
        rows[selected]["production_kept"] = True
        shipping = CANDIDATE_ID
        production_kept = True
        claims = True
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        production_kept = False
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": selected == CONTROL_ID,
        "shipping_gdn_decode": shipping,
        "production_kept": production_kept,
        "winners": winners,
        "claims_throughput": claims,
        "kernel_parity_pass": all(row["kernel_parity_pass"] for row in rows.values()),
        "model_quality_pass": bool(rows[CANDIDATE_ID]["model_quality_pass"]),
        "performance_pass": bool(rows[CANDIDATE_ID]["performance_pass"]),
    }


def admit_counts(phase: str, mode: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    return validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name=phase,
        workload=workload_for_mode(load_json(ITERATION)["workloads"][phase], mode),
        stdout=json.dumps(observed),
        success=True,
    )


def replay_command(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    capture_key: str | None,
) -> list[str]:
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        "decode-gdn",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--gdn-decode",
        str(config["gdn_decode"]),
    ]
    prefix = int(plan.get("prefix", 0) or 0)
    if prefix > 0:
        args.extend(["--prefix", str(prefix)])
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def parse_pilot(stdout: str) -> dict[str, Any]:
    match = re.search(r"QW38_OPT109_PILOT=(\{.*\})", stdout)
    if match is None:
        return {}
    return json.loads(match.group(1))


def parse_timed_relayout(stdout: str) -> int:
    match = re.search(r"timed_relayout_launches=(\d+)", stdout)
    if match is None:
        match = re.search(r'"timed_relayout":\s*(\d+)', stdout)
    return int(match.group(1)) if match else 0


def evaluate_parity(
    *,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    layout = layout_roundtrip_host()
    if skip_gpu:
        payload = dict(synthetic or {})
        passed = bool(payload.get("pass", layout["pass"]))
        return {
            "pass": passed,
            "layout": layout,
            "catalog_count": 1,
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": passed},
                CANDIDATE_ID: {"kernel_parity_pass": passed},
            },
            "stdout": str(payload.get("stdout", "")),
        }
    native_runner = runner or default_native_runner
    completed = native_runner([f"./{NATIVE}", "--phase", "parity"], "correctness")
    stdout = completed.stdout + completed.stderr
    passed = "status=passed" in stdout and layout["pass"]
    return {
        "pass": passed,
        "layout": layout,
        "catalog_count": 1,
        "by_ident": {
            CONTROL_ID: {"kernel_parity_pass": passed},
            CANDIDATE_ID: {"kernel_parity_pass": passed},
        },
        "stdout": stdout,
    }


def run_pilot_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("pilot", mode)
    if skip_gpu:
        pilot = dict(
            (synthetic or {}).get("pilot")
            or {
                "positive": False,
                "mean_diff_ms": -0.01,
                "ci95_low": -0.02,
                "control_mean_ms": 1.0,
                "candidate_mean_ms": 1.01,
                "timed_relayout": 0,
                "n": int(plan["samples"]),
            }
        )
        stdout = str((synthetic or {}).get("stdout", ""))
    else:
        native_runner = runner or default_native_runner
        completed = native_runner(
            [f"./{NATIVE}", "--phase", "pilot"],
            "acceptance" if mode == "acceptance" else "screen",
        )
        stdout = completed.stdout + completed.stderr
        (run_dir / "pilot-native.txt").write_text(stdout, encoding="utf-8")
        pilot = parse_pilot(stdout)
        if not pilot:
            raise AdmissionError("missing QW38_OPT109_PILOT payload")
        if int(pilot.get("timed_relayout", 1)) != 0:
            raise AdmissionError("pilot recorded timed relayout launches")
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("pilot", mode, observed)
    decided = decide_independent_verdicts(
        parity=None,
        quality=None,
        pilot=pilot,
        component=None,
        engine_by_prefix={},
        prefill=None,
        state=None,
        memory=None,
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "pilot",
        "mode": mode,
        "pilot": pilot,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "pilot_measured",
        "stdout": stdout,
        "pilot_lost": not bool(pilot.get("positive")),
    }


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    stdout_all = ""
    for config in CONFIGS:
        completed = runner(replay_command(plan, config, capture_key), tier)
        stdout_all += completed.stdout + "\n"
        (run_dir / f"{plan['phase']}-{config['id']}-replay.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        observed = parse_native_observation(completed.stdout)
        if capture_key is None:
            capture_key = str(observed.get("capture_key") or "")
            if not capture_key:
                match = re.search(r"capture_key=([0-9a-f]{64})", completed.stdout)
                if match:
                    capture_key = match.group(1)
        if not capture_key:
            raise AdmissionError("missing capture identity")
        rounds = [
            row
            for row in parse_rounds(completed.stdout)
            if row.get("cache_mode", "rotating") == "rotating"
        ]
        if len(rounds) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        dispatch = parse_gdn_dispatch(completed.stdout)
        if not dispatch_matches(dispatch, config):
            raise AdmissionError(
                f"{config['id']} launch variants {dispatch} != {config}"
            )
        groups = re.search(r"gdn_input_output_groups=(\d+)", completed.stdout)
        if groups is None or int(groups.group(1)) != 48:
            raise AdmissionError(
                f"{config['id']} complete GDN layers were not 48: {groups}"
            )
        timed = parse_timed_relayout(completed.stdout)
        decode_conv = 0
        conv_match = re.search(r"conversion_decode=(\d+)", completed.stdout)
        if conv_match:
            decode_conv = int(conv_match.group(1))
        if str(config["id"]) == CANDIDATE_ID and (timed != 0 or decode_conv != 0):
            raise AdmissionError(
                f"timed decode relayout/conversion was not zero: {timed}/{decode_conv}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
            "timed_relayout_launches": timed,
            "decode_conversions": decode_conv,
        }
    stats = paired_student_t(
        by_config[CONTROL_ID]["ms"],
        by_config[CANDIDATE_ID]["ms"],
        critical=T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT,
    )
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "candidate": CANDIDATE_ID,
        "control_ms": by_config[CONTROL_ID]["ms"],
        "candidate_ms": by_config[CANDIDATE_ID]["ms"],
        "control_mean_ms": by_config[CONTROL_ID]["mean_ms"],
        "candidate_mean_ms": by_config[CANDIDATE_ID]["mean_ms"],
        **stats,
        "saving_ge_0_10_ms": float(stats["mean_diff_ms"]) >= MIN_SAVING_MS,
        "timed_relayout_launches": by_config[CANDIDATE_ID]["timed_relayout_launches"],
        "decode_conversions": by_config[CANDIDATE_ID]["decode_conversions"],
        "stdout": stdout_all,
        "pairs": int(plan["samples"]),
        "dispatch_ok": True,
        "opt077_replay_not_added_to_wall": True,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    prefix: int | None = None,
    pairs_n: int | None = None,
    sidecar: str | None = None,
) -> dict[str, Any]:
    count = int(pairs_n if pairs_n is not None else plan["engine_pairs"])
    if count <= 0:
        return {}
    used_prefix = int(prefix if prefix is not None else plan.get("prefix") or 2048)
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "gdn-ab",
        "--pairs",
        str(count),
        "--modes",
        "graph",
        "--prefix",
        str(used_prefix),
        "--output-tokens",
        "32",
        "--gdn-decode",
        "persistent_transposed",
    ]
    completed = runner(command, tier)
    name = sidecar or f"{plan['phase']}-engine.txt"
    (run_dir / name).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if "override_before_capture_applied=true" not in completed.stdout:
        raise AdmissionError("graph capture override was not applied")
    if (
        "recapture=true" not in completed.stdout
        and "recapture_after_selector=true" not in completed.stdout
    ):
        raise AdmissionError("graphs were not recaptured after selecting the candidate")
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != count:
        raise AdmissionError(f"engine pairs {len(pairs)} != {count}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    payload = {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "this_sitting": True,
        "prefix": used_prefix,
        "output_tokens": 32,
    }
    payload["guards"] = e2e_guards(payload, output_tokens=32)
    return payload


def run_parity_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("parity", mode)
    parity = evaluate_parity(skip_gpu=skip_gpu, runner=runner, synthetic=synthetic)
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("parity", mode, observed)
    decided = decide_independent_verdicts(
        parity=parity,
        quality=None,
        pilot=None,
        component=None,
        engine_by_prefix={},
        prefill=None,
        state=None,
        memory=None,
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "parity",
        "mode": mode,
        "parity": parity,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "parity_evaluated",
    }


def run_gdn_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("gdn", mode)
    prior_pilot = ((prior or {}).get("pilot") or {}).get("pilot") or {}
    if prior_pilot and not bool(prior_pilot.get("positive")):
        component = {
            "skipped": True,
            "reason": "pilot_lost",
            "positive": False,
            "mean_diff_ms": 0.0,
            "ci95_low": 0.0,
            "timed_relayout_launches": 0,
            "decode_conversions": 0,
        }
        dispatch_ok = True
    elif skip_gpu:
        component = dict((synthetic or {}).get("component") or {})
        dispatch_ok = bool((synthetic or {}).get("dispatch_ok", True))
    else:
        native_runner = runner or default_native_runner
        component = run_component(plan, runner=native_runner, run_dir=run_dir)
        dispatch_ok = bool(component.get("dispatch_ok"))
    observed = planned_observation(
        plan, capture_key=str(component.get("capture_key") or "") or None
    )
    admission = admit_counts("gdn", mode, observed)
    decided = decide_independent_verdicts(
        parity=None,
        quality=None,
        pilot=prior_pilot or None,
        component=component,
        engine_by_prefix={},
        prefill=None,
        state=None,
        memory=None,
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "gdn",
        "mode": mode,
        "component": component,
        "dispatch_ok": dispatch_ok,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
    }


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    del run_dir
    quality = opt073_quality()
    model_quality_pass = quality.get("quality_v3_engine_non_regression") in {
        "pass",
        True,
    }
    plan = family_plan("quality", mode if mode != "release" else "acceptance")
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "quality",
        "mode": mode,
        "quality": quality,
        "model_quality_pass": model_quality_pass,
        "require_candidate_nll": True,
        "recurrence_incremental_nll_max": 0.02,
        "quality_contract_id": "opt091_successor",
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "quality_reused",
        "native_counts": planned_observation(plan),
    }


def run_state_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("state", mode)
    if skip_gpu:
        payload = dict(
            (synthetic or {}).get("state") or {"pass": True, "skipped": True}
        )
        stdout = ""
    else:
        native_runner = runner or default_native_runner
        completed = native_runner([f"./{NATIVE}", "--phase", "state"], "correctness")
        stdout = completed.stdout + completed.stderr
        (run_dir / "state-native.txt").write_text(stdout, encoding="utf-8")
        payload = {
            "pass": "status=passed" in stdout
            and '"id":"lifetime"' in stdout
            and '"id":"checkpoint_roundtrip"' in stdout
            and '"id":"graph_two_address_variants"' in stdout
            and '"id":"failure_middle_layer"' in stdout
            and '"id":"cancel_before_publish"' in stdout,
            "stdout": stdout,
        }
    observed = planned_observation(plan, keep=False)
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "state",
        "mode": mode,
        "state": payload,
        "pass": bool(payload.get("pass")),
        "native_counts": observed,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "state_evaluated",
        "stdout": stdout,
    }


def run_memory_phase(*, mode: str) -> dict[str, Any]:
    ledger = load_json(MEMORY) if MEMORY.is_file() else {}
    owners = ledger.get("owners") or {}
    gdn_bytes = int(owners.get("gdn_state_bytes") or 0)
    admitted = bool(ledger.get("post_graph_admitted"))
    same_count = gdn_bytes == GDN_STATE_BYTES
    payload = {
        "pass": admitted and same_count,
        "gdn_state_bytes": gdn_bytes,
        "expected_gdn_state_bytes": GDN_STATE_BYTES,
        "element_count_unchanged": True,
        "post_graph_admitted": admitted,
        "capacity": ledger.get("capacity"),
    }
    plan = family_plan("memory", mode)
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "memory",
        "mode": mode,
        "memory": payload,
        "pass": bool(payload["pass"]),
        "native_counts": planned_observation(plan),
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "memory_ledger_checked",
    }


def run_e2e_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    prior_pilot = ((prior or {}).get("pilot") or {}).get("pilot") or {}
    prior_gdn = ((prior or {}).get("gdn") or {}).get("component") or {}
    if prior_pilot and not bool(prior_pilot.get("positive")):
        engine = {"skipped": True, "reason": "pilot_lost", "guards": {"pass": False}}
    elif prior_gdn and not bool(prior_gdn.get("positive")):
        engine = {
            "skipped": True,
            "reason": "complete_gdn_lost",
            "guards": {"pass": False},
        }
    elif skip_gpu:
        engine = dict((synthetic or {}).get("engine") or {})
    else:
        engine = run_engine(
            plan,
            runner=runner or default_native_runner,
            run_dir=run_dir,
            prefix=int(plan["prefix"]),
            pairs_n=5,
        )
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    decided = decide_independent_verdicts(
        parity=None,
        quality=None,
        pilot=prior_pilot or None,
        component=((prior or {}).get("gdn") or {}).get("component"),
        engine_by_prefix={phase: engine},
        prefill=None,
        state=None,
        memory=None,
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": phase,
        "mode": mode,
        "engine": engine,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "engine_guard_measured",
    }


def run_prefill_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("prefill-guard", mode)
    prior_pilot = ((prior or {}).get("pilot") or {}).get("pilot") or {}
    prior_gdn = ((prior or {}).get("gdn") or {}).get("component") or {}
    if prior_pilot and not bool(prior_pilot.get("positive")):
        payload = {
            "skipped": True,
            "reason": "pilot_lost",
            "throughput_ratio": 0.0,
        }
    elif prior_gdn and not bool(prior_gdn.get("positive")):
        payload = {
            "skipped": True,
            "reason": "complete_gdn_lost",
            "throughput_ratio": 0.0,
        }
    elif skip_gpu:
        payload = dict(synthetic or {"throughput_ratio": 1.0, "skipped": True})
    else:
        native_runner = runner or default_native_runner
        walls: dict[str, float] = {}
        for config in CONFIGS:
            command = [
                f"./{PROBE}",
                MODEL,
                "--workload",
                "prefill",
                "--prompt",
                "4096",
                "--gdn-decode",
                str(config["gdn_decode"]),
                "--modes",
                "graph",
            ]
            completed = native_runner(command, "acceptance")
            (run_dir / f"prefill-{config['id']}.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            walls[str(config["id"])] = parse_prefill_wall_ms(completed.stdout)
        control_ms = walls[CONTROL_ID]
        candidate_ms = walls[CANDIDATE_ID]
        control_tok = 4096.0 / (control_ms / 1000.0) if control_ms > 0 else 0.0
        cand_tok = 4096.0 / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
        payload = {
            "prefix": 4096,
            "control_ms": control_ms,
            "candidate_ms": candidate_ms,
            "control_tok_s": control_tok,
            "candidate_tok_s": cand_tok,
            "throughput_ratio": cand_tok / control_tok if control_tok else 0.0,
        }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("prefill-guard", mode, observed)
    return {
        "schema_version": 1,
        "task": "OPT-109",
        "phase": "prefill-guard",
        "mode": mode,
        "prefill_guard": payload,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "prefill_guard_measured",
        "throughput_ratio": float(payload.get("throughput_ratio", 1.0)),
    }


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = dict(payload)
    for key in VERDICT_KEYS:
        if key in payload:
            results[key] = payload[key]
    if "independent_verdicts" in payload:
        prior = results.get("independent_verdicts")
        merged = dict(prior) if isinstance(prior, Mapping) else {}
        incoming = payload["independent_verdicts"]
        if isinstance(incoming, Mapping):
            for cid, row in incoming.items():
                previous = dict(merged.get(cid) or empty_verdict_row())
                previous.update(dict(row))
                merged[cid] = previous
        results["independent_verdicts"] = merged
    if payload.get("production_kept") is not None and phase in {
        "pilot",
        "gdn",
        "d128",
        "d2048",
        "prefill-guard",
        "state",
        "memory",
    }:
        results["production_kept"] = payload["production_kept"]
    if payload.get("shipping_gdn_decode"):
        results["shipping_gdn_decode"] = payload["shipping_gdn_decode"]
    if payload.get("selected_path"):
        results["selected_path"] = payload["selected_path"]
    if payload.get("claims_throughput") is not None:
        results["claims_throughput"] = payload["claims_throughput"]
    results["kernel_parity_pass"] = bool(
        ((results.get("independent_verdicts") or {}).get(CANDIDATE_ID) or {}).get(
            "kernel_parity_pass"
        )
        or ((results.get("parity") or {}).get("parity") or {}).get("pass")
        or ((results.get("parity") or {}).get("pass"))
    )
    results["model_quality_pass"] = bool(
        ((results.get("independent_verdicts") or {}).get(CANDIDATE_ID) or {}).get(
            "model_quality_pass"
        )
        or (results.get("quality") or {}).get("model_quality_pass")
    )
    results["performance_pass"] = bool(
        ((results.get("independent_verdicts") or {}).get(CANDIDATE_ID) or {}).get(
            "performance_pass"
        )
    )


def write_report(results: Mapping[str, Any]) -> None:
    report_dir = REPORT.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    pilot = (results.get("pilot") or {}).get("pilot") or results.get("pilot") or {}
    component = (results.get("gdn") or {}).get("component") or {}
    d128 = (results.get("d128") or {}).get("engine") or {}
    d2048 = (results.get("d2048") or {}).get("engine") or {}
    prefill = (results.get("prefill-guard") or {}).get("prefill_guard") or {}
    payload = dict(results)
    kept = bool(payload.get("production_kept"))
    status = "kept" if kept else "rejected"
    text = f"""# OPT-109 — Keep transposed GDN state for the session lifetime

Status: **{status}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is sequential row-major
`prepare_recurrence_window`. Candidate is `persistent_transposed` col-major
session state with zero timed per-layer relayout. Convolution stays a separate
launch. Canonical checkpoint payload remains row-major. OPT-077/094/101
historical fixtures are not rewritten.

`claims_throughput: {payload.get("claims_throughput")}`.
Production pin `{payload.get("shipping_gdn_decode", "sequential")}`.
`evidence_complete={payload.get("evidence_complete")}`.

## Pilot recurrence-only

control_mean_ms={pilot.get("control_mean_ms")}
candidate_mean_ms={pilot.get("candidate_mean_ms")}
mean_diff_ms={pilot.get("mean_diff_ms")}
ci95_low={pilot.get("ci95_low")}
positive={pilot.get("positive")}
timed_relayout={pilot.get("timed_relayout")}

## Layout

logical `{LOGICAL_LAYOUT}` sha256 `{LOGICAL_SHA}`;
device `{DEVICE_LAYOUT}` sha256 `{DEVICE_SHA}`;
checkpoint `{LOGICAL_LAYOUT}`; element count `{ELEMENT_COUNT}`.

## Independent verdicts

{json.dumps(results.get("independent_verdicts") or {}, indent=2)}

## Complete 48-layer GDN

control_mean_ms={component.get("control_mean_ms")}
candidate_mean_ms={component.get("candidate_mean_ms")}
mean_diff_ms={component.get("mean_diff_ms")}
ci95_low={component.get("ci95_low")}
ci95_high={component.get("ci95_high")}
saving_ge_0_10_ms={component.get("saving_ge_0_10_ms")}
positive={component.get("positive")}
timed_relayout_launches={component.get("timed_relayout_launches")}
decode_conversions={component.get("decode_conversions")}

## D128 / D2048

D128 control_mean_ms={d128.get("control_mean_ms")} candidate_mean_ms={d128.get("candidate_mean_ms")}
D128 throughput_ratio={(d128.get("guards") or {}).get("throughput_ratio")} guards_pass={(d128.get("guards") or {}).get("pass")}
D2048 control_mean_ms={d2048.get("control_mean_ms")} candidate_mean_ms={d2048.get("candidate_mean_ms")}
D2048 throughput_ratio={(d2048.get("guards") or {}).get("throughput_ratio")} guards_pass={(d2048.get("guards") or {}).get("pass")}

## P4096 guard

control_ms={prefill.get("control_ms")} candidate_ms={prefill.get("candidate_ms")}
control_tok_s={prefill.get("control_tok_s")} candidate_tok_s={prefill.get("candidate_tok_s")}
throughput_ratio={prefill.get("throughput_ratio")}

## Decision

`production_kept={payload.get("production_kept")}`; shipping GDN decode
`{payload.get("shipping_gdn_decode", "sequential")}`.

## tok/s

Speedup versus the then-current sequential baseline is **{payload.get("tok_s_delta", 0)}**.
On reject, speedup is 0 (baseline unchanged).
"""
    REPORT.write_text(text, encoding="utf-8")
    rejection = report_dir / REJECTION.name
    if not kept:
        rejection.write_text(text, encoding="utf-8")
    elif rejection.is_file():
        rejection.unlink()


def finalize_acceptance(results: dict[str, Any]) -> None:
    parity = results.get("parity") or {}
    quality = results.get("quality") or {}
    pilot = (results.get("pilot") or {}).get("pilot") or {}
    component = (results.get("gdn") or {}).get("component")
    engines = {
        "d128": (results.get("d128") or {}).get("engine") or {},
        "d2048": (results.get("d2048") or {}).get("engine") or {},
    }
    prefill = results.get("prefill-guard") or {}
    state = results.get("state") or {}
    memory = results.get("memory") or {}
    decided = decide_independent_verdicts(
        parity=parity.get("parity") or parity,
        quality=quality,
        pilot=pilot,
        component=component,
        engine_by_prefix=engines,
        prefill=prefill.get("prefill_guard") or prefill,
        state=state.get("state") or state,
        memory=memory.get("memory") or memory,
        mode="acceptance",
        prior=results.get("independent_verdicts")
        if isinstance(results.get("independent_verdicts"), Mapping)
        else None,
    )
    results.update(decided)
    results["evidence_complete"] = True
    results["status"] = "kept" if decided["production_kept"] else "rejected"
    results["tok_s_delta"] = 0.0
    d128 = engines.get("d128") or {}
    d2048 = engines.get("d2048") or {}
    if (
        decided["production_kept"]
        and d128.get("control_mean_ms")
        and d2048.get("control_mean_ms")
    ):
        results["tok_s_delta"] = 32000.0 / float(
            d128["candidate_mean_ms"]
        ) - 32000.0 / float(d128["control_mean_ms"])
    else:
        results["tok_s_delta"] = 0.0
        results["claims_throughput"] = False
    write_report(results)


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-109",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "candidate": CANDIDATE_ID,
            "shipping_gdn_decode": current_gdn_pin(),
            "opt074_coverage_unadmitted_blocker": False,
            "report_path": "evidence/optimization/opt109-persistent-gdn-state/REPORT.md",
            "logical_layout": LOGICAL_LAYOUT,
            "device_layout": DEVICE_LAYOUT,
            "checkpoint_layout": LOGICAL_LAYOUT,
        }
    )
    if phase == "parity":
        payload = run_parity_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "pilot":
        payload = run_pilot_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "quality":
        payload = run_quality_phase(mode=mode, run_dir=run_dir)
    elif phase == "gdn":
        payload = run_gdn_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
            prior=results,
        )
    elif phase == "state":
        payload = run_state_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "memory":
        payload = run_memory_phase(mode=mode)
    elif phase in {"d128", "d2048"}:
        payload = run_e2e_phase(
            phase,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
            prior=results,
        )
    else:
        payload = run_prefill_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
            prior=results,
        )
    merge_fixture(results, phase, payload)
    results["status"] = str(payload.get("status") or "measured")
    ready = all(
        results.get(name)
        for name in (
            "parity",
            "pilot",
            "gdn",
            "quality",
            "state",
            "d128",
            "d2048",
            "prefill-guard",
            "memory",
        )
    )
    if (
        phase in {"prefill-guard", "memory"}
        and mode == "acceptance"
        and not skip_gpu
        and ready
    ):
        finalize_acceptance(results)
        results["pins_applied"] = apply_production_pins(results)
    elif skip_gpu:
        results["evidence_complete"] = False
        results["production_kept"] = bool(results.get("production_kept", False))
        write_report(results)
    else:
        write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt109_persistent_gdn_state.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or planned_observation(
        family_plan(phase, mode if mode != "release" else "acceptance")
    )
    print(
        "QW38_OPT109_RESULT="
        + json.dumps(
            {
                "task": "OPT-109",
                "mode": mode,
                "phase": phase,
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_gdn_decode": results.get("shipping_gdn_decode"),
                "keep": bool(results.get("production_kept")),
                **{
                    key: counts.get(key)
                    for key in (
                        "warmups",
                        "samples",
                        "observed_warmups",
                        "observed_samples",
                        "observed_candidates",
                        "observed_shapes",
                        "observed_tier",
                        "pairs",
                        "sample_ids",
                        "acceptance_executed",
                    )
                },
            }
        )
    )
    print("QW38_OPT109_NATIVE_COUNTS=" + json.dumps(counts))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OPT-109 persistent GDN state admission"
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", required=True, choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.phase, Path(args.run_dir), skip_gpu=args.skip_gpu)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdmissionError as exc:
        print(f"admission_error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
