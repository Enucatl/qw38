from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from tools.compare_llama_authority import compare, load_rows, parse_fields

ROOT = Path(__file__).resolve().parents[1]


def test_llama_authority_contract_pins_build_identity_and_raw_tokens() -> None:
    contract = json.loads(
        (ROOT / "pins" / "llama_authority_contract.json").read_text(encoding="utf-8")
    )
    artifact_pins = json.loads(
        (ROOT / "pins" / "artifacts.lock.json").read_text(encoding="utf-8")
    )
    assert (
        contract["source"]["revision"]
        == artifact_pins["sources"]["llama_cpp"]["revision"]
    )
    assert contract["model"]["sha256"] == artifact_pins["model"]["sha256"]
    assert contract["diagnostic"]["input_tokens"] == [42, 3649]
    assert contract["diagnostic"]["template_case"] == "user_no_thinking"
    assert contract["build"]["cmake"]["CMAKE_CUDA_ARCHITECTURES"] == "120"
    assert "qw38-llama-token-oracle" in contract["build"]["targets"]


def test_beginner_authority_documentation_keeps_proof_limits_visible() -> None:
    handbook = (ROOT / "docs" / "36-independent-llama-authority.md").read_text(
        encoding="utf-8"
    )
    for concept in (
        "Three authorities, three jobs",
        "primary semantic authority",
        "independent same-GGUF",
        "Why this test starts with raw token IDs",
        "Complete logit rows",
        "Reporting is not admission",
        "Failure boundaries",
        "Proof boundary",
    ):
        assert concept in handbook


def test_checked_in_llama_authority_retains_identity_and_full_row_metrics() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "llama_scalar_authority.json").read_text(encoding="utf-8")
    )
    assert fixture["admission_status"] == "reporting_only_tolerances_not_frozen"
    assert fixture["input_tokens"] == [42, 3649]
    assert fixture["template_identity"]["exact_equal"] is True
    assert fixture["template_identity"]["token_ids"] == [
        248045,
        846,
        198,
        9419,
        248046,
        198,
        248045,
        74455,
        198,
        248068,
        271,
        248069,
        271,
    ]
    assert [row["llama_greedy_token"] for row in fixture["comparisons"]] == [
        3649,
        1277,
    ]
    assert [row["quartz_greedy_token"] for row in fixture["comparisons"]] == [
        3649,
        1277,
    ]
    assert all(row["metrics"]["count"] == 248320 for row in fixture["comparisons"])
    assert all(
        row["metrics"]["expected_non_finite"]["nan"] == 0
        and row["metrics"]["actual_non_finite"]["nan"] == 0
        for row in fixture["comparisons"]
    )


def test_authority_field_protocol_rejects_duplicates(tmp_path: Path) -> None:
    fields = tmp_path / "fields.txt"
    fields.write_text("token_count=2\ntoken_count=3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        parse_fields(fields)


def test_authority_field_protocol_ignores_container_banner(tmp_path: Path) -> None:
    fields = tmp_path / "fields.txt"
    fields.write_text("==========\ntoken_count=2\n", encoding="utf-8")

    assert parse_fields(fields) == {"token_count": "2"}


def test_authority_rows_require_exact_byte_count(tmp_path: Path) -> None:
    logits = tmp_path / "logits.bin"
    logits.write_bytes(struct.pack("<3f", 1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="expected exactly 16"):
        load_rows(logits, 2, 2)


def test_same_gguf_comparison_retains_full_row_metrics(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "diagnostic": {"input_tokens": [42, 7]},
                "model": {"sha256": "a" * 64},
                "source": {"revision": "b" * 40},
            }
        ),
        encoding="utf-8",
    )
    native_fields = tmp_path / "native.txt"
    llama_fields = tmp_path / "llama.txt"
    common = "token_count=2\nvocabulary_size=3\ntoken_0=42\ntoken_1=7\n"
    native_fields.write_text(common + "greedy_0=1\ngreedy_1=2\n", encoding="utf-8")
    llama_fields.write_text(
        "llama_version=pinned\n" + common + "greedy_0=1\ngreedy_1=2\n",
        encoding="utf-8",
    )
    native_logits = tmp_path / "native.bin"
    llama_logits = tmp_path / "llama.bin"
    native_logits.write_bytes(struct.pack("<6f", 0, 2, 1, 0, 1, 3))
    llama_logits.write_bytes(struct.pack("<6f", 0, 2, 1, 0, 1, 2.5))

    result = compare(
        native_logits,
        llama_logits,
        native_fields,
        llama_fields,
        contract,
    )

    assert result["admission_status"] == "reporting_only_tolerances_not_frozen"
    comparisons = result["comparisons"]
    assert isinstance(comparisons, list)
    assert comparisons[0]["metrics"]["passed"] is True
    assert comparisons[1]["metrics"]["maximum_absolute_error"] == 0.5
