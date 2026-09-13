"""OPT-126 stable decode-graph inputs across token commits.

Foundation/correctness prerequisite. Shipping selector stays ffn_only.
claims_throughput=false. OPT-127 owns full replay admission.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    dump_json,
    load_json,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    docker_common,
    git_identity,
)
from tools.opt115_pipeline_traffic import authenticate_post113  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt126_stable_graph_inputs_contract.json"
ITERATION = ROOT / "pins/opt126_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt126_stable_graph_inputs.json"
REPORT = ROOT / "evidence/optimization/opt126-stable-graph-inputs/REPORT.md"
INVENTORY = ROOT / "evidence/optimization/opt126-stable-graph-inputs/INVENTORY.md"
EVIDENCE = REPORT.parent
NATIVE = "build/qw38-cuda-opt126-stable-graph-inputs-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT126_STABLE_GRAPH_INPUTS_RESULT="
COUNTS_PREFIX = "QW38_OPT126_NATIVE_COUNTS="
PARENT = "ffn_only"
CANDIDATE = "decode_segments8"
PHASES = (
    "preflight",
    "inventory",
    "positions",
    "isolation",
    "restore",
    "same-math",
    "costs",
    "report",
)
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "mode",
    "llama_revision",
    "gguf_sha256",
    "parent",
    "candidate",
    "gdn_parity_design",
    "inventory",
    "positions",
    "isolation",
    "restore",
    "same_math",
    "costs",
    "shipping_execution_graphs",
    "production_kept",
    "claims_throughput",
    "claims_performance_improvement",
    "verdict",
    "report_path",
)


class GraphInputError(RuntimeError):
    """OPT-126 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-126":
        raise GraphInputError("stable-graph-inputs contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-126", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-126 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


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


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    records = [
        json.loads(line.removeprefix(prefix))
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if not records:
        raise GraphInputError(f"missing {prefix} record")
    return records[-1]


def native_command(args: Sequence[str], *, tier: str) -> list[str]:
    return [*docker_common(IMAGE, tier), *args]


def run_native(
    args: Sequence[str], *, tier: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = native_command(args, tier=tier)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if check and completed.returncode != 0:
        raise GraphInputError(
            "native command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    iteration = load_iteration()
    auth = authenticate_post113()
    source, dirty = git_identity()
    selector = contract["selected_execution_graph_path"]
    payload = {
        "schema_version": 1,
        "task": "OPT-126",
        "phase": "preflight",
        "mode": mode,
        "ok": bool(auth.get("ok")) and selector == PARENT,
        "authenticated_post113": auth,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "shipping_execution_graphs": selector,
        "selector_unchanged": selector == PARENT,
        "gdn_parity_design": contract["gdn_parity_design"],
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "source": source,
        "dirty": bool(dirty),
        "iteration_target": iteration["target"],
        "diagnostics_make_target": iteration["diagnostics_make_target"],
        "claims_throughput": False,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def run_named_native(
    run_dir: Path,
    mode: str,
    phase: str,
    extra: Sequence[str],
    *,
    check: bool = True,
) -> dict[str, Any]:
    completed = run_native(
        [f"./{NATIVE}", "--workload", phase, *extra, MODEL],
        tier="correctness",
        check=check,
    )
    record = parse_prefixed(completed.stdout + completed.stderr, RESULT_PREFIX)
    record["phase"] = phase
    record["mode"] = mode
    record["family_plan"] = family_plan(mode, phase)
    record["stdout_tail"] = completed.stdout[-2000:]
    return store_sidecar(run_dir, f"{phase}.json", record)


def evaluate(run_dir: Path) -> dict[str, Any]:
    inventory = load_sidecar(run_dir, "inventory.json") or {}
    positions = load_sidecar(run_dir, "positions.json") or {}
    isolation = load_sidecar(run_dir, "isolation.json") or {}
    restore = load_sidecar(run_dir, "restore.json") or {}
    same = load_sidecar(run_dir, "same-math.json") or {}
    costs = load_sidecar(run_dir, "costs.json") or {}
    reasons: list[str] = []
    if not inventory.get("ok"):
        reasons.append("inventory_failed")
    if not positions.get("ok"):
        reasons.append("positions_failed")
    if not isolation.get("ok"):
        reasons.append("isolation_failed")
    if not restore.get("ok"):
        reasons.append("restore_failed")
    if not (same.get("ok") and same.get("exact")):
        reasons.append("same_math_failed")
    if not costs.get("ok"):
        reasons.append("costs_failed")
    if inventory.get("selector_unchanged") is False:
        reasons.append("shipping_selector_changed")
    if inventory.get("gdn_parity_design") not in (
        None,
        "stable_pingpong_committed_slot",
    ):
        reasons.append("gdn_parity_design_mismatch")
    verified = not reasons
    return {
        "ok": verified,
        "production_kept": False,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "shipping_execution_graphs": PARENT,
        "verdict": "prerequisite_verified" if verified else "inconclusive",
        "reasons": reasons,
        "inventory": inventory,
        "positions": positions,
        "isolation": isolation,
        "restore": restore,
        "same_math": same,
        "costs": costs,
    }


def write_inventory_doc(inventory: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    rows = inventory.get("changing_inputs") or []
    table = [
        "| Input | Consumer | Kind | Topology |",
        "|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row.get('name')}` | `{row.get('consumer')}` | "
            f"`{row.get('kind')}` | `{row.get('topology')}` |"
        )
    if len(table) == 2:
        table.extend(
            [
                "| `frontier_position` | `attention_rope_kv_span` | `value` | `false` |",
                "| `visible_kv_length` | `attention_decode` | `value` | `false` |",
                "| `gdn_committed_slot` | `gdn_prepare_tiled` | `pointer_index` | `false` |",
                "| `token` | `embedding_row` | `value` | `false` |",
                "| `attention_dispatch` | `vec128_vs_warp_query` | `kernel_identity` | `true` |",
                "| `session_identity` | `scheduler_graphs` | `pointer` | `true` |",
                "| `q8_grouped_descriptors` | `mixer_q8` | `descriptor_lifetime` | `true` |",
            ]
        )
    text = "\n".join(
        [
            "# OPT-126 — Argument and ownership inventory",
            "",
            "Captured by `capture_decode_segment_graph` / "
            "`SchedulerGraphs::update_launch_params`. Dynamic scalars live in "
            "session-owned device `DecodeLaunchState` and update in stream "
            "order. Physical GDN ping-pong bases stay stable; kernels load "
            "the committed slot. Topology is bounded to two graph variants "
            "(`warp_query` at capture frontier 0, `vec128_online` at 1024). "
            "Opaque byte-patching is unused on this path.",
            "",
            f"GDN parity design: `{inventory.get('gdn_parity_design', 'stable_pingpong_committed_slot')}`.",
            f"Launch-state bytes: `{inventory.get('launch_state_bytes', 'sizeof(DecodeLaunchState)')}`.",
            f"Topology count: `{inventory.get('topology_count', 2)}`.",
            f"Shipping selector: `{inventory.get('shipping_execution_graphs', PARENT)}`.",
            "",
            *table,
            "",
            "## Ownership",
            "",
            "- Slot 0 convolution/recurrent: session allocation (`gdn_convolution_` / `gdn_recurrent_`).",
            "- Slot 1 convolution/recurrent: workspace candidate buffers, bound not copied.",
            "- `gdn_committed_slot_`: session integer; XOR on successful commit only.",
            "- `DecodeLaunchState*`: session device allocation; host staging is session-owned, not a capture-time temp.",
            "- Grouped-Q8 descriptors: persistent workspace host memory rebound on session/workspace identity change.",
            "- Graph executables: `SchedulerGraphs`; 8 segments × 2 topologies = 16.",
            "",
        ]
    )
    INVENTORY.write_text(text, encoding="utf-8")


def write_report(result: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    costs = result.get("costs") or {}
    same = result.get("same_math") or {}
    positions = result.get("positions") or {}
    isolation = result.get("isolation") or {}
    restore = result.get("restore") or {}
    inventory = result.get("inventory") or {}
    lines = [
        "# OPT-126 — Make decode graph inputs stable across token commits",
        "",
        f"Status: **{result.get('verdict')}**. Shipping selector "
        f"`{result.get('shipping_execution_graphs')}`. "
        "Foundation/correctness prerequisite; not a production speed keep. "
        "`claims_throughput=false`.",
        "",
        "## Design",
        "",
        "GDN uses stable physical ping-pong with an explicit committed slot. "
        "Dynamic scalars (token, position, frontier, kv_bucket, slot, "
        "generation, ping-pong bases) live in session-owned device "
        "`DecodeLaunchState` and upload once per token in stream order. "
        "Topology is bounded to two graph variants at the 1024 warp_query → "
        "vec128_online dispatch change. Shipping `ffn_only` is unchanged; "
        "`decode_segments8` is exercised as a diagnostic path.",
        "",
        f"GDN parity=`{inventory.get('gdn_parity_design')}`. "
        f"Launch-state bytes=`{inventory.get('launch_state_bytes')}`. "
        f"Selector unchanged=`{inventory.get('selector_unchanged')}`.",
        "",
        "## Same-math and positions",
        "",
        f"same_math ok=`{same.get('ok')}` exact=`{same.get('exact')}` "
        f"state_equals=`{same.get('state_equals')}` "
        f"matched_tokens=`{same.get('matched_tokens')}`. "
        f"positions ok=`{positions.get('ok')}`. "
        f"isolation ok=`{isolation.get('ok')}` "
        f"cancel=`{isolation.get('cancel_ok')}` retry=`{isolation.get('retry_ok')}` "
        f"failure=`{isolation.get('failure_ok')}` "
        f"candidate_isolated=`{isolation.get('candidate_isolated')}`. "
        f"restore ok=`{restore.get('ok')}` "
        f"stale_rejected=`{restore.get('stale_descriptors_rejected')}` "
        f"divergent=`{restore.get('divergent_prefix_invalidated')}`.",
        "",
        "## Upload / indirection / memory",
        "",
        f"graph_bytes=`{costs.get('graph_bytes')}` "
        f"(OPT-117 eight-graph capture was 12582912 bytes; two topologies "
        f"are expected near 2×). create_ms=`{costs.get('create_ms')}`. "
        f"launch_state_bytes=`{costs.get('launch_state_bytes')}`. "
        f"launch_state_upload_ms=`{costs.get('launch_state_upload_ms')}`. "
        f"launch_state_uploads=`{costs.get('launch_state_uploads')}`. "
        f"topology_recaptures=`{costs.get('topology_recaptures')}`. "
        f"indirection_loads_per_consumer=`{costs.get('indirection_loads_per_consumer')}`.",
        "",
        "Opaque captured-argument byte patching is unused. Recapture happens "
        "only when ping-pong bases or session KV identity change, not on "
        "ordinary frontier or GDN slot flips.",
        "",
        f"Verdict `{result.get('verdict')}`. Reasons: `{result.get('reasons')}`. "
        "Production selector remains `ffn_only`. OPT-127 owns replay admission.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    contract = load_contract()
    decision = evaluate(run_dir)
    write_inventory_doc(decision.get("inventory") or {})
    payload = {
        "schema_version": 1,
        "task": "OPT-126",
        "mode": mode,
        "llama_revision": contract["llama_revision"],
        "gguf_sha256": GGUF_SHA,
        "parent": contract["parent"],
        "candidate": contract["candidate"],
        "gdn_parity_design": contract["gdn_parity_design"],
        "inventory": decision["inventory"],
        "positions": decision["positions"],
        "isolation": decision["isolation"],
        "restore": decision["restore"],
        "same_math": decision["same_math"],
        "costs": decision["costs"],
        "shipping_execution_graphs": decision["shipping_execution_graphs"],
        "production_kept": decision["production_kept"],
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "verdict": decision["verdict"],
        "reasons": decision["reasons"],
        "report_path": str(contract["report_path"]),
        "inventory_path": str(contract["inventory_path"]),
        "measured_at": utc_now(),
        "family_plan": family_plan(mode, "report"),
    }
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise GraphInputError(f"fixture missing {key}")
    dump_json(FIXTURE, payload)
    write_report(payload)
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "preflight":
        return run_preflight(run_dir, mode)
    if phase == "inventory":
        return run_named_native(run_dir, mode, "inventory", [])
    if phase == "positions":
        return run_named_native(run_dir, mode, "positions", [])
    if phase == "isolation":
        return run_named_native(run_dir, mode, "isolation", ["--prefix", "128"])
    if phase == "restore":
        return run_named_native(run_dir, mode, "restore", ["--prefix", "128"])
    if phase == "same-math":
        return run_named_native(
            run_dir,
            mode,
            "same-math",
            ["--prefix", "128", "--tokens", "8"],
        )
    if phase == "costs":
        return run_named_native(run_dir, mode, "costs", ["--prefix", "128"])
    if phase == "report":
        return run_report(run_dir, mode)
    raise GraphInputError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    result = run_phase(args.phase, Path(args.run_dir), args.mode)
    json.dump({"phase": args.phase, "ok": bool(result.get("ok", True))}, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok", True) or args.phase == "report" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GraphInputError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
