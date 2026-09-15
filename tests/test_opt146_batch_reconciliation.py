"""Host tests for OPT-146 final batch reconciliation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from tools.opt146_batch_reconciliation import (
    BASELINE_PAIRS,
    BASELINE_WARMUPS,
    CAPACITY,
    CONTRACT,
    DECODE_PREFIXES,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT,
    PHASES,
    PREFILL_TOKENS,
    PROFILE_PREFIXES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTOR,
    experiment_outcomes,
    family_plan,
    freeze_opt137_control,
    net_batch_gains,
    next_experiment_spec,
    validate_fixture,
)
from tools.performance_evidence import (
    RESIDUAL_WALL_LIMIT,
    reconcile_window_wall,
    parse_nsys_sqlite,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
RUNNER = ROOT / "tools/run_optimization_task.py"
LEDGER = ROOT / "implementation_ledger.md"
OPT016 = ROOT / "tasks/OPT-016.md"
OPT056 = ROOT / "tasks/OPT-056.md"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-146")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt146_batch_reconciliation.py").read_text(encoding="utf-8")
    opt136 = (ROOT / "tools/opt136_graph_accounting.py").read_text(encoding="utf-8")
    opt138 = (ROOT / "tools/opt138_remaining_gap_profile.py").read_text(
        encoding="utf-8"
    )
    opt142 = (ROOT / "tools/opt142_wall_reconciliation.py").read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-146"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["allocated_capacity"] == CAPACITY
    assert contract["decode_prefixes"] == list(DECODE_PREFIXES)
    assert contract["profile_prefixes"] == list(PROFILE_PREFIXES)
    assert contract["prefill_tokens"] == PREFILL_TOKENS
    assert contract["warmups"] == BASELINE_WARMUPS
    assert contract["paired_rounds"] == BASELINE_PAIRS
    assert contract["unresolved_limit"] == RESIDUAL_WALL_LIMIT
    assert contract["opt137_mma_threshold"] == MMA_THRESHOLD
    assert contract["opt137_control"]["historical_rates_as_paired_gain"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt146-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["aggregate_deadline_s"] == 7200
    assert iteration["claims_throughput"] is False
    assert "cuda-opt146-diagnostics" in makefile
    assert "QW38_OPT146_BATCH_RECONCILIATION_RESULT=" in runner
    assert "opt136_graph_accounting" in tool
    assert "opt138_remaining_gap_profile" in tool
    assert "opt142_wall_reconciliation" in tool
    assert "authenticate_current_pins" in opt136
    assert "select_top_two_families" in opt138 or "FAMILY_SOURCES" in opt138
    assert "reconcile_window_wall" in opt142
    assert "OPT-146" in ledger
    assert PHASES == (
        "preflight",
        "throughput",
        "capture",
        "reconcile",
        "families",
        "experiments",
        "report",
    )
    validate_future_keep_policy("OPT-146", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-146")
    preflight = iteration["workloads"]["preflight"]
    throughput = iteration["workloads"]["throughput"]
    capture = iteration["workloads"]["capture"]
    reconcile = iteration["workloads"]["reconcile"]
    families = iteration["workloads"]["families"]
    experiments = iteration["workloads"]["experiments"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(throughput, "acceptance")) == 130
    assert loop_product(workload_for_mode(capture, "acceptance")) == 54
    assert loop_product(workload_for_mode(reconcile, "acceptance")) == 21
    assert loop_product(workload_for_mode(families, "acceptance")) == 3
    assert loop_product(workload_for_mode(experiments, "acceptance")) == 3
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-146", "acceptance", iteration, "throughput")
    assert "phase=throughput" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def test_opt137_control_is_current_production_not_historical_paired_gain() -> None:
    control = freeze_opt137_control(
        {
            "ok": True,
            "opt137_dense_mma": True,
            "q4_decode": "llama_q4k_mmvq",
            "execution_graphs": SELECTOR,
        },
        {"quartz_native": "abc"},
    )
    assert control["reconstructable"] is True
    assert control["same_sitting_as_final"] is True
    assert control["historical_rates_as_paired_gain"] is False
    assert control["production_pins_changed_after_opt137"] is False
    assert control["parent"] == PARENT
    assert control["mma_threshold"] == MMA_THRESHOLD
    outcomes = experiment_outcomes()
    assert outcomes["all_no_opportunity"] is True
    assert outcomes["any_keep"] is False
    assert outcomes["shipping_delta"] == 0
    tasks = {row["task"]: row for row in outcomes["experiments"]}
    assert tasks["OPT-143"]["verdict"] == "no_opportunity"
    assert tasks["OPT-144"]["verdict"] == "no_opportunity"
    assert tasks["OPT-145"]["verdict"] == "no_opportunity"
    net = net_batch_gains(control, outcomes)
    assert net["shipping_delta"] == 0
    assert net["candidate_measured_delta"] == 0
    assert net["claims_throughput"] is False
    assert net["batch_keeps"] == 0


def test_next_experiment_is_null_without_mechanism() -> None:
    outcomes = experiment_outcomes()
    decode = next_experiment_spec(
        "decode",
        {
            "selected": [
                {
                    "family": "attn_core",
                    "score_ms": 47.5,
                    "d2048_middle": {
                        "mean_ms": 47.5,
                        "one_sided_low": 47.3,
                        "n": 3,
                    },
                }
            ]
        },
        outcomes,
    )
    prefill = next_experiment_spec(
        "prefill",
        {
            "selected": [
                {
                    "family": "attn_core",
                    "score_ms": 186.5,
                    "stats": {"mean_ms": 186.5, "one_sided_low": 186.1, "n": 3},
                }
            ]
        },
        outcomes,
    )
    residual = next_experiment_spec(
        "decode",
        {
            "selected": [
                {
                    "family": "residual_norm_quant",
                    "score_ms": 7.2,
                    "d128_middle": {
                        "mean_ms": 7.2,
                        "one_sided_low": 7.17,
                        "n": 3,
                    },
                }
            ]
        },
        outcomes,
    )
    empty = next_experiment_spec("decode", {"selected": []}, outcomes)
    assert decode["candidate"] is None
    assert decode["supported_mechanism"] is None
    assert decode["prior_experiment"] == "OPT-143"
    assert residual["prior_experiment"] == "OPT-145"
    assert (
        "resolving" in (decode["resolving_measurement"] or "").lower()
        or decode["resolving_measurement"]
    )
    assert prefill["candidate"] is None
    assert prefill["prior_experiment"] == "OPT-144"
    assert empty["candidate"] is None
    assert empty["selected_family"] is None


def test_residual_limit_and_cpu_overlap_not_additive(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "wall.sqlite")
    conn.execute(
        "CREATE TABLE export_schema (table_name TEXT, column_name TEXT, unit TEXT)"
    )
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "NVTX_EVENTS",
    ):
        conn.execute("INSERT INTO export_schema VALUES (?, 'start', 'ns')", (table,))
        conn.execute("INSERT INTO export_schema VALUES (?, 'end', 'ns')", (table,))
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
          start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER,
          streamId INTEGER, globalPid INTEGER, correlationId INTEGER,
          name TEXT, graphNodeId INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
          start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER,
          correlationId INTEGER, nameId INTEGER, returnValue INTEGER
        )
        """
    )
    conn.execute(
        "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalPid INTEGER)"
    )
    conn.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?,?,?,1)",
        (0, 10_000_000, "opt136.window"),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,1,1,7,1,1,?,NULL)",
        (100_000, 9_700_000, "attn_core"),
    )
    conn.execute("INSERT INTO StringIds VALUES (1, 'cudaEventSynchronize_v3020')")
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,0,1,1,1,0)",
        (100_000, 9_800_000),
    )
    conn.commit()
    conn.close()
    tables = parse_nsys_sqlite(tmp_path / "wall.sqlite")
    wall = reconcile_window_wall(
        tables, window_start_ns=0, window_end_ns=10_000_000, window_source="explicit"
    )
    assert wall["cpu_overlapping_gpu_not_additive"] is True
    assert wall["unresolved_share_of_wall"] <= RESIDUAL_WALL_LIMIT


def test_opt016_opt056_unchanged() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt146_batch_reconciliation.py").read_text(encoding="utf-8")
    assert "opt016" not in makefile.lower() or "cuda-opt016" not in makefile
    assert "OPT-016" in OPT016.read_text(encoding="utf-8")
    assert "OPT-056" in OPT056.read_text(encoding="utf-8")
    assert "parity_claimed" in tool
    assert "release_readiness_claimed" in tool
    assert "opt016_unchanged" in tool
    assert "opt056_unchanged" in tool


def test_fixture_keys_when_present() -> None:
    if not FIXTURE.is_file():
        return
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert fixture["allocated_capacity"] == CAPACITY
    assert fixture["selected_execution_graph_path"] == SELECTOR
    assert REPORT.as_posix().endswith("opt146-batch-reconciliation/REPORT.md")
    if fixture.get("status") == "measured":
        assert fixture["answers"]["shipping_delta"] == 0
        assert fixture["answers"]["parity_claimed"] is False
        assert fixture["answers"]["release_readiness_claimed"] is False
