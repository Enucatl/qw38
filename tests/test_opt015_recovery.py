from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt015_recovery_contract.json"
FIXTURE = ROOT / "fixtures/opt015_recovery.json"
ATTRIBUTION = ROOT / "fixtures/cuda_prefill_attribution.json"
LLAMA_BENCH = (
    ROOT / "evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json"
)
CATEGORIES = (
    "embedding",
    "gdn",
    "attention",
    "ffn_mmq",
    "logits",
    "commit_sync",
    "graph",
    "other_idle",
)
RANKED = (
    "q4k_q6k_mma_mmq",
    "gdn_fused_token_loop",
    "causal_mma_attention",
    "optional_2048_prompt_graphs",
)
REQUIRED_PHRASES = (
    "thousands of tok/s",
    "ds4 cannot run this Qwen GGUF",
    "no ds4 same-model baseline",
    "MIT",
    "sparse",
    "compressed attention",
    "plan.md",
    "not a throughput gate",
    "not llama.cpp parity",
    "cc83d7b",
    *CATEGORIES,
    *RANKED,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _llama_2k(records: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in records if row.get("n_prompt") == 2048]
    assert matches, "scaling JSON is missing the 2048-token llama-bench object"
    return matches[0]


def _section_after(text: str, heading: str, stop: str | None) -> str:
    start = text.find(heading)
    assert start >= 0, f"missing heading {heading!r}"
    body = text[start + len(heading) :]
    if stop is None:
        return body
    end = body.find(stop)
    assert end >= 0, f"missing stop heading {stop!r}"
    return body[:end]


def validate_result(result: Any) -> None:
    contract = _load(CONTRACT)
    attribution = _load(ATTRIBUTION)
    llama = _llama_2k(_load(LLAMA_BENCH))
    assert isinstance(result, dict)
    assert set(result) == set(contract["required_fixture_keys"])
    assert result["schema_version"] == 1
    assert result["task"] == "OPT-015"
    assert result["status"] == "source_inspected"
    assert result["status"] != "measured"
    assert result["llama_revision"] == contract["llama_revision"]
    assert result["dwarfstar_pin"] == contract["dwarfstar_pin"]
    assert result["ds4_license"] == "MIT"
    assert result["ds4_can_run_qwen_gguf"] is False
    assert result["ds4_same_model_baseline"] is False
    assert contract["opt016_in_scope"] is False
    assert contract["ds4_can_run_qwen_gguf"] is False
    assert contract["ds4_same_model_baseline"] is False
    assert result["report_path"] == contract["report_path"]
    proof = result["proof_limit"]
    assert isinstance(proof, str)
    proof_cf = proof.casefold()
    for phrase in contract["proof_limit"]:
        assert phrase.casefold() in proof_cf
    assert "not a throughput gate" in proof_cf
    assert result["categories"] == list(CATEGORIES) == contract["categories"]
    assert result["ranked_recovery"] == list(RANKED) == contract["ranked_recovery"]
    assert result["ranked_recovery"][0] == "q4k_q6k_mma_mmq"
    assert result["non_transferable"] == contract["non_transferable"]
    assert "compressed_attention" in result["non_transferable"]
    assert "compressed_attention" not in result["transferable"]
    assert isinstance(result["transferable"], list) and result["transferable"]
    quartz = result["quartz"]
    assert quartz["prompt_tokens"] == 2048 == contract["primary_tokens"]
    assert quartz["measurement_utc"] == attribution["measurement_utc"]
    assert quartz["wall_ms"] == pytest.approx(attribution["wall_ms"], abs=1e-6)
    assert quartz["tok_s"] == pytest.approx(attribution["tok_s"], abs=1e-6)
    assert set(quartz["categories_ms"]) == set(CATEGORIES)
    for name in CATEGORIES:
        assert quartz["categories_ms"][name] == pytest.approx(
            attribution["categories_ms"][name], abs=1e-6
        )
    llama_cpp = result["llama_cpp"]
    assert llama_cpp["avg_ts"] == pytest.approx(llama["avg_ts"], abs=1e-6)
    assert llama_cpp["avg_ns"] == llama["avg_ns"]
    assert llama_cpp["n_prompt"] == 2048
    assert llama_cpp["n_ubatch"] == 512 == llama["n_ubatch"]
    assert llama_cpp["flash_attn"] == llama["flash_attn"]
    assert llama_cpp["build_commit"] == llama["build_commit"] == "cc83d7b"
    assert llama_cpp["test_time"] == llama["test_time"]
    ffn_ms = float(quartz["categories_ms"]["ffn_mmq"])
    ceiling = 2048.0 / (ffn_ms / 1000.0)
    assert result["gap"]["ffn_only_tok_s_ceiling"] == pytest.approx(ceiling, abs=1e-6)
    assert result["gap"]["ratio"] == pytest.approx(
        float(llama_cpp["avg_ts"]) / float(quartz["tok_s"]), abs=1e-6
    )
    category_map = result["category_map"]
    assert set(category_map) == set(CATEGORIES)
    for name in CATEGORIES:
        entry = category_map[name]
        assert set(entry) == {"path", "rank"}
        assert isinstance(entry["path"], str) and entry["path"]
        assert entry["rank"] is None or isinstance(entry["rank"], int)
    assert category_map["ffn_mmq"]["path"] == "q4k_q6k_mma_mmq"
    assert category_map["ffn_mmq"]["rank"] == 1
    assert category_map["graph"]["path"] in {
        "optional_2048_prompt_graphs",
        "keep",
    }
    assert 1 in {category_map[name]["rank"] for name in CATEGORIES}


def test_opt015_contract_and_fixture_are_connected() -> None:
    contract = _load(CONTRACT)
    fixture = _load(FIXTURE)
    assert contract["schema_version"] == 1
    assert contract["task"] == "OPT-015"
    assert contract["primary_tokens"] == 2048
    assert contract["quartz_source"] == "fixtures/cuda_prefill_attribution.json"
    assert contract["llama_source"] == (
        "evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json"
    )
    assert contract["llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    assert contract["ds4_role"] == "technique_inspiration_only"
    assert contract["nsight"] == "not_required"
    assert contract["opt016_in_scope"] is False
    validate_result(fixture)


def test_opt015_report_contains_required_headings_and_phrases() -> None:
    contract = _load(CONTRACT)
    report_path = ROOT / contract["report_path"]
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "placeholder" not in text.casefold()
    for heading in contract["report_headings"]:
        assert heading in text
    folded = text.casefold()
    for phrase in REQUIRED_PHRASES:
        assert phrase.casefold() in folded
    for label in ("Measured", "External", "Estimated", "Proposed"):
        assert f"**{label}**" in text
    citations = (
        "mmq.cuh",
        "mmq-config-ampere.cuh",
        "gated_delta_net.cu",
        "fattn.cu",
        "qwen35.cpp",
        "ds4.c",
        "cuda_prefill_attribution.json",
        "llama-bench-prefill-2k-8k-32k.json",
        "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "c1d4597a80e300b803dc642519718f2c999589da",
        "c238077a87186381bf626cc531bccffe1fef79e7",
    )
    for citation in citations:
        assert citation in text


def test_opt015_compressed_attention_is_non_transferable() -> None:
    contract = _load(CONTRACT)
    text = (ROOT / contract["report_path"]).read_text(encoding="utf-8")
    transferable = _section_after(
        text,
        "## Transferable methods",
        "## Non-transferable ds4 model policies",
    )
    non_transferable = _section_after(
        text,
        "## Non-transferable ds4 model policies",
        "## Quartz 2K category map",
    )
    transfer_cf = transferable.casefold()
    non_cf = non_transferable.casefold()
    assert "compressed attention" in non_cf
    assert "sparse" in non_cf
    assert "compressed attention" not in transfer_cf
    assert "sparse" not in transfer_cf


def test_opt015_validator_rejects_inadmissible_evidence() -> None:
    fixture = _load(FIXTURE)

    def drop_ffn_mmq(payload: dict[str, Any]) -> None:
        payload["category_map"].pop("ffn_mmq")

    def ds4_baseline(payload: dict[str, Any]) -> None:
        payload["ds4_same_model_baseline"] = True

    def ds4_can_run(payload: dict[str, Any]) -> None:
        payload["ds4_can_run_qwen_gguf"] = True

    def compressed_transferable(payload: dict[str, Any]) -> None:
        payload["non_transferable"].remove("compressed_attention")
        payload["transferable"].append("compressed_attention")

    def remove_rank_1(payload: dict[str, Any]) -> None:
        payload["ranked_recovery"].pop(0)
        payload["category_map"]["ffn_mmq"]["rank"] = 2

    def wrong_avg_ts(payload: dict[str, Any]) -> None:
        payload["llama_cpp"]["avg_ts"] = 1.0

    def drop_throughput_gate(payload: dict[str, Any]) -> None:
        payload["proof_limit"] = payload["proof_limit"].replace(
            "not a throughput gate", ""
        )

    def mark_measured(payload: dict[str, Any]) -> None:
        payload["status"] = "measured"

    mutations = []
    for mutate in (
        drop_ffn_mmq,
        ds4_baseline,
        ds4_can_run,
        compressed_transferable,
        remove_rank_1,
        wrong_avg_ts,
        drop_throughput_gate,
        mark_measured,
    ):
        changed = deepcopy(fixture)
        mutate(changed)
        mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(AssertionError):
            validate_result(mutation)
