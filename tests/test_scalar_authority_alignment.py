from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.compare_scalar_authorities import gdn_permute
from tools.freeze_scalar_tolerances import cosine_floor, nice_ceiling, tap_key

ROOT = Path(__file__).resolve().parents[1]


def test_gdn_layout_normalization_round_trips() -> None:
    grouped = tuple(float(value) for value in range(48 * 3))
    tiled = gdn_permute(grouped, 3, grouped_to_tiled=True)
    assert gdn_permute(tiled, 3, grouped_to_tiled=False) == grouped
    assert tiled[:9] == grouped[:3] + grouped[9:12] + grouped[18:21]


def test_frozen_margin_rounding_is_deterministic() -> None:
    assert nice_ceiling(15.828716278076172) == 18.0
    assert nice_ceiling(0.0) == 0.0
    assert cosine_floor(0.999567917498981) == 0.999524
    assert cosine_floor(1.0) == 1.0


def test_every_scalar_authority_sample_passes_all_frozen_gates() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "scalar_authority_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    tolerances = json.loads(
        (ROOT / "pins" / "scalar_oracle_tolerances.json").read_text(encoding="utf-8")
    )
    assert fixture["admission_status"] == "admitted_by_frozen_scalar_tolerances"
    assert len(fixture["comparisons"]) == 194
    counts = {"official_vs_quartz": 0, "llama_vs_quartz": 0}
    for row in fixture["comparisons"]:
        for side in counts:
            if side not in row:
                continue
            counts[side] += 1
            metrics = row[side]
            gate = tolerances["authorities"][side]["taps"][tap_key(row)]["admission"]
            assert metrics["maximum_absolute_error"] <= gate["maximum_absolute_error"]
            assert (
                metrics["root_mean_square_error"]
                <= gate["maximum_root_mean_square_error"]
            )
            assert metrics["cosine_similarity"] >= gate["minimum_cosine_similarity"]
            for participant in ("expected_non_finite", "actual_non_finite"):
                assert metrics[participant] == {
                    "nan": 0,
                    "negative_infinity": 0,
                    "positive_infinity": 0,
                }
    assert counts == {"official_vs_quartz": 194, "llama_vs_quartz": 156}


def test_frozen_authority_identity_and_greedy_policy_are_exact() -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "scalar_authority_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    identities = fixture["identities"]
    assert identities["model_gguf_sha256"] == (
        "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
    )
    assert identities["blobs"] == {
        "llama_cpp_sha256": "e950c76b04580d251ba2a9da5a0ba21cb73135202201f1ebd0696066ef0dc245",
        "official_taps_sha256": "99d47367f411786f4d5f483a0a927491e412eca119bf7d7dcf0805538b1ab164",
        "quartz_sha256": "96e14a3e29af2781a9a716ec913098f2b576d988e27ab9ff5d8c3ab548261b17",
    }
    assert len(identities["local_sources"]) == 8
    for name, expected in identities["local_sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    tolerances = json.loads(
        (ROOT / "pins" / "scalar_oracle_tolerances.json").read_text(encoding="utf-8")
    )
    freezer = ROOT / "tools" / "freeze_scalar_tolerances.py"
    assert (
        tolerances["calibration"]["freezer_sha256"]
        == hashlib.sha256(freezer.read_bytes()).hexdigest()
    )
    assert fixture["greedy"]["near_tie_exceptions"] == []
    for authority in ("official", "quartz", "llama_cpp"):
        assert [row["greedy_token"] for row in fixture["greedy"][authority]] == [
            3649,
            1277,
        ]
        assert all(row["margin"] > 0.0 for row in fixture["greedy"][authority])


def test_scalar_trace_contract_labels_convolution_state_channel_major() -> None:
    contract = json.loads(
        (ROOT / "pins" / "scalar_trace_contract.json").read_text(encoding="utf-8")
    )
    by_name = {tap["name"]: tap for tap in contract["layer_taps"]}
    assert by_name["gdn.convolution_state"]["shape"] == [10240, 4]
    for required in ("gdn.value", "gdn.decay", "gdn.beta", "gdn.gated_output"):
        assert required in by_name


def test_beginner_tolerance_chapter_preserves_proof_limits() -> None:
    chapter = (ROOT / "docs" / "38-scalar-authority-tolerances.md").read_text(
        encoding="utf-8"
    )
    for concept in (
        "Comparable boundaries",
        "Runtime-private boundaries",
        "Why layout normalization is not a tolerance",
        "Absolute error",
        "RMS error",
        "Cosine similarity",
        "How the frozen gates were chosen",
        "Greedy equality and near-ties",
        "What this proves",
        "What this does not prove",
    ):
        assert concept in chapter
