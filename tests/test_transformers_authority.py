from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_transformers_authority import verify_file

ROOT = Path(__file__).resolve().parents[1]


def test_transformers_contract_pins_every_authority_input() -> None:
    contract = json.loads(
        (ROOT / "pins" / "transformers_authority_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["checkpoint"]["revision"] == (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert contract["transformers"]["revision"] == (
        "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
    )
    assert len(contract["checkpoint"]["shards"]) == 18
    assert contract["checkpoint"]["tensor_count"] == 1199
    assert contract["execution"]["attention"] == "eager"
    assert contract["execution"]["input_tokens"] == [42, 3649]


def test_transformers_file_identity_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "shard.safetensors"
    artifact.write_bytes(b"authority")
    digest = hashlib.sha256(b"authority").hexdigest()
    verify_file(artifact, 9, digest)

    with pytest.raises(ValueError, match="size differs"):
        verify_file(artifact, 8, digest)
    with pytest.raises(ValueError, match="hash differs"):
        verify_file(artifact, 9, "0" * 64)


def test_transformers_fixture_retains_complete_tap_manifest() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "transformers_scalar_authority.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["admission_status"] == "reporting_only_tolerances_not_frozen"
    assert fixture["execution"]["input_tokens"] == [42, 3649]
    assert fixture["execution"]["greedy_tokens"] == [3649, 1277]
    assert fixture["checkpoint"]["loaded_text_wrapper_tensor_count"] == 1184
    assert fixture["checkpoint"]["ignored_mtp_tensor_count"] == 15
    assert fixture["logits"]["official_sha256"] == (
        "9b64105a1c7262271c85054ef30cd116e0af4e85a497e6ae24c478007ed97947"
    )

    records = fixture["taps"]["records"]
    assert len(records) == 238
    assert sum(record["bytes"] for record in records) == fixture["taps"]["bytes"]
    assert [record["offset"] for record in records] == [
        sum(previous["bytes"] for previous in records[:index])
        for index in range(len(records))
    ]
    assert all(
        record["summary"]["finite_count"] == record["summary"]["count"]
        for record in records
    )
    by_name = {record["name"]: record for record in records}
    assert by_name["position.0.layer.0.gdn.recurrent_state"]["shape"] == [
        1,
        48,
        128,
        128,
    ]
    assert by_name["position.1.layer.3.attention.key_cache"]["shape"] == [
        1,
        4,
        2,
        256,
    ]
    assert by_name["position.1.logits"]["shape"] == [248320]


def test_transformers_comparison_is_reporting_only() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "transformers_scalar_authority.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["official_greedy_token"] for row in fixture["comparisons"]] == [
        3649,
        1277,
    ]
    for row in fixture["comparisons"]:
        assert row["quartz"]["passed"] is False
        assert row["llama_cpp"]["passed"] is False
        assert row["quartz"]["expected_non_finite"]["nan"] == 0
        assert row["llama_cpp"]["expected_non_finite"]["nan"] == 0


def test_beginner_transformers_authority_documentation_has_proof_boundary() -> None:
    handbook = (ROOT / "docs" / "37-transformers-authority.md").read_text(
        encoding="utf-8"
    )
    for concept in (
        "Original checkpoint and GGUF are different artifacts",
        "What a Safetensors shard is",
        "What eager execution means",
        "GPU, CPU, and disk offload",
        "Hooks and taps",
        "Why raw token IDs come first",
        "What this proves",
        "What this does not prove",
    ):
        assert concept in handbook
