"""Host tests for OPT-129 matched Quartz vs llama decode-attention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt129_matched_attention import (
    ATTENTION_LAYERS,
    CONTRACT,
    FIXTURE,
    ITERATION,
    LAYERS,
    LLAMA_REV,
    MATERIALITY,
    NATIVE,
    PHASES,
    PREFIXES,
    PROVENANCE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    family_plan,
    inspect_sources,
    llama_padded_nkv,
    llama_selected_kernel,
    quartz_shipping_kernel,
    rank_opt130,
    validate_fixture,
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
NATIVE_SRC = ROOT / "cuda/opt129_matched_attention_test.cu"
ADAPTER = ROOT / "cuda/opt129_llama_fattn_adapter.cuh"
PIN = ROOT / "cuda/attention_decode_path.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-129")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    pin = PIN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-129"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["crossover_threshold"] == 1024
    assert contract["prefixes"] == list(PREFIXES)
    assert contract["attention_layers"] == list(LAYERS)
    assert iteration["diagnostics_make_target"] == "cuda-opt129-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt129-diagnostics" in makefile
    assert "qw38-cuda-opt129-matched-attention-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "identity|inspect|replay|numerical" in native
    assert "QW38_OPT129_MATCHED_ATTENTION_RESULT=" in native
    assert "QW38_OPT129_NATIVE_COUNTS=" in native
    assert "QW38_OPT129_MATCHED_ATTENTION_RESULT=" in runner
    assert "flash_attn_ext_vec_f16" in adapter
    assert "convert_physical_bf16_to_llama_f16" in adapter
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in pin
    assert PHASES == ("inspect", "replay", "numerical", "ranking", "report")
    provenance = _json(PROVENANCE)
    assert provenance["llama_revision"] == LLAMA_REV
    assert provenance["license"] == "MIT"
    validate_future_keep_policy("OPT-129", iteration)


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-129")
    inspect = iteration["workloads"]["inspect"]
    replay = iteration["workloads"]["replay"]
    numerical = iteration["workloads"]["numerical"]
    ranking = iteration["workloads"]["ranking"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(inspect, "feedback")) == 1
    assert loop_product(workload_for_mode(replay, "feedback")) == 432
    assert loop_product(workload_for_mode(numerical, "feedback")) == 6
    assert loop_product(workload_for_mode(ranking, "feedback")) == 1
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-129", "feedback", iteration, "inspect")
    assert "phase=inspect" in described
    assert "hybrid_crossover@1024" in " ".join(iteration["proof_limit"])
    plan = family_plan("feedback", "inspect")
    assert "task=OPT-129" in plan
    assert "phase=inspect" in plan


def test_dispatch_identities_and_inspect_sources() -> None:
    assert quartz_shipping_kernel(128) == "warp_query_decode_attention"
    assert quartz_shipping_kernel(1023) == "warp_query_decode_attention"
    assert quartz_shipping_kernel(1024) == "vec128_online_decode_attention"
    assert quartz_shipping_kernel(2048) == "vec128_online_decode_attention"
    assert quartz_shipping_kernel(4096) == "vec128_online_decode_attention"
    assert quartz_shipping_kernel(8192) == "warp_query_decode_attention"
    assert llama_padded_nkv(129) == 256
    assert llama_padded_nkv(1024) == 1024
    assert llama_padded_nkv(1025) == 1280
    assert llama_selected_kernel(256) == "flash_attn_ext_vec<256,1>"
    assert llama_selected_kernel(2048) == "flash_attn_ext_vec<256,1>"
    assert llama_selected_kernel(8192) == "fattn-mma-f16_ncols1=1_ncols2=8"
    host = inspect_sources()
    assert host["hybrid_crossover_unchanged"] is True
    assert host["crossover_threshold"] == 1024
    assert host["has_best_fattn_kernel"] is True
    assert host["prefixes"]["1024"]["llama_vec_is_selected"] is True
    assert host["prefixes"]["8192"]["llama_vec_is_selected"] is False


def test_ranking_caps_two_transfers_and_materiality() -> None:
    contract = _json(CONTRACT)
    tiny = rank_opt130(
        [
            {
                "prefix": 128,
                "quartz_enclosing_ms": 0.01,
                "matched_bf16_enclosing_ms": 0.01,
                "adapter_ms": 0.001,
            },
            {
                "prefix": 2048,
                "quartz_enclosing_ms": 0.01,
                "matched_bf16_enclosing_ms": 0.01,
                "adapter_ms": 0.001,
            },
        ],
        contract,
    )
    assert tiny["attention_material"] is False
    assert tiny["d2048_branch_ms"] == 0.01 * ATTENTION_LAYERS
    assert tiny["d2048_share_of_request"] < MATERIALITY
    assert len(tiny["transfers"]) <= 2
    heavy = rank_opt130(
        [
            {
                "prefix": 2048,
                "quartz_enclosing_ms": 0.08,
                "matched_bf16_enclosing_ms": 0.04,
                "adapter_ms": 0.01,
            },
            {
                "prefix": 8192,
                "quartz_enclosing_ms": 0.4,
                "matched_bf16_enclosing_ms": 0.2,
                "adapter_ms": 0.05,
            },
            {
                "prefix": 32768,
                "quartz_enclosing_ms": 1.2,
                "matched_bf16_enclosing_ms": 0.6,
                "adapter_ms": 0.2,
            },
        ],
        contract,
    )
    assert heavy["attention_material"] is True
    assert len(heavy["transfers"]) <= 2
    ids = [row["id"] for row in heavy["transfers"]]
    assert "occupancy_partition_or_gqa_kv_reuse" in ids
    assert "llama_mma_decode_consumer" in ids
    assert heavy["opt130_disposition"] == "proceed"
    assert tiny["opt130_disposition"] == "no_opportunity"
    fixture = {
        "schema_version": 1,
        "task": "OPT-129",
        "status": "host",
        "measurement_utc": "2026-09-14T00:00:00Z",
        "identity": {},
        "kernel_identities": {},
        "replay_inputs": {},
        "per_shape": [],
        "numerical": {},
        "adapter_cost": {},
        "ranking": {"transfers": heavy["transfers"][:2]},
        "attention_material": heavy["attention_material"],
        "production_kept": True,
        "crossover_threshold": 1024,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "report_path": str(REPORT),
    }
    assert validate_fixture(fixture)["ok"] is True
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert FIXTURE.name == "opt129_matched_attention.json"
