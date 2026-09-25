"""Verify the frozen Q4_K screen rejects mismatched inputs and artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "q4k_screen", Path(__file__).parents[1] / "scripts/q4k_development_screen.py"
)
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def identities() -> dict[str, dict]:
    """Return a valid pair with only the 192 MLP quantizers changed."""
    baseline = {
        "precision_policy_id": 1026,
        "compiler": {"ident": "qw38-candidate-v1-policy-frozen"},
        "manifest_digest": "a" * 64,
        "source_hash": "b" * 64,
        "config_hash": "c" * 64,
        "tokenizer_hash": "d" * 64,
        "tensor_count": 869,
        "storage_counts": {"int4_grouped": 200, "int8_grouped": 210},
        "decode_dispatch": {
            "activation_policy": "bf16",
            "quantized_kernel": "grouped_gemv",
            "fallback": None,
        },
        "logical_quantizer_counts": {
            "none": 459,
            "q4_g64_v0": 8,
            "q8_g32_v0": 0,
            "q4_g64_candidate_v1": 192,
            "q8_g32_candidate_v1": 210,
        },
    }
    candidate = copy.deepcopy(baseline)
    candidate["precision_policy_id"] = 1027
    candidate["manifest_digest"] = "e" * 64
    candidate["compiler"]["ident"] = (
        "qw38-candidate-v2-q4k-llama-e6ab7c1a4-no-imatrix-bf16-operands"
    )
    candidate["logical_quantizer_counts"]["q4_g64_candidate_v1"] = 0
    candidate["logical_quantizer_counts"]["q4_k_candidate_v2"] = 192
    return {"q4g64": baseline, "q4k": candidate}


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Prepare eight manifest-bound synthetic development windows."""
    base, out = tmp_path / "base", tmp_path / "out"
    (base / "tokens").mkdir(parents=True)
    monkeypatch.setattr(screen, "BASE", base)
    monkeypatch.setattr(screen, "OUT", out)
    windows, cases, scores = (
        [],
        [],
        ["id\tprompt_sha256\ttarget_sha256\tprompt_tokens\ttarget_tokens\tnll"],
    )
    for i in range(screen.COUNT):
        case_id = f"validation.window-{i:03}"
        data = np.zeros(512, dtype="<u4").tobytes()
        path = base / "tokens" / f"{case_id}.u32le"
        path.write_bytes(data)
        prompt_hash = hashlib.sha256(data[:1536]).hexdigest()
        target_hash = hashlib.sha256(data[1536:]).hexdigest()
        windows.append(
            {
                "tokens_file": path.name,
                "tokens_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        cases.append(f"{case_id}\t{path}\t{prompt_hash}\t{target_hash}")
        scores.append(f"{case_id}\t{prompt_hash}\t{target_hash}\t384\t128\t128")
    (base / "tokens/validation.manifest.json").write_text(
        json.dumps({"role": "development-screening", "windows": windows})
    )
    (base / "development-cases.tsv").write_text("\n".join(cases) + "\n")
    (base / "gpu-existing-q4-k-m-scores.tsv").write_text("\n".join(scores) + "\n")
    screen.prepare()
    return json.loads((out / "screen.json").read_text())


def test_valid_identities() -> None:
    """Accept old identity records without an explicit zero Q4_K count."""
    screen.validate_identities(identities())


@pytest.mark.parametrize(
    "field",
    [
        "source_hash",
        "config_hash",
        "tokenizer_hash",
        "manifest_digest",
        "precision_policy_id",
    ],
)
def test_wrong_artifact(field: str) -> None:
    """Reject source drift, identical artifacts, and swapped policy arms."""
    pair = identities()
    pair["q4k"][field] = (
        pair["q4g64"][field]
        if field in ("manifest_digest", "precision_policy_id")
        else "f" * 64
    )
    with pytest.raises(ValueError):
        screen.validate_identities(pair)


@pytest.mark.parametrize(
    "field,value",
    [("q8_g32_candidate_v1", 209), ("q4_k_candidate_v2", 191), ("none", 458)],
)
def test_changed_weight_inventory(field: str, value: int) -> None:
    """Reject an incomplete MLP conversion or another changed weight family."""
    pair = identities()
    pair["q4k"]["logical_quantizer_counts"][field] = value
    with pytest.raises(ValueError):
        screen.validate_identities(pair)


def test_changed_arithmetic() -> None:
    """Reject even matched arms that no longer use BF16 operands."""
    pair = identities()
    for arm in pair.values():
        arm["decode_dispatch"]["activation_policy"] = "fp32"
    with pytest.raises(ValueError):
        screen.validate_identities(pair)


@pytest.mark.parametrize("suffix", ["prompt", "target", "mask"])
def test_changed_case_file(frozen: dict, suffix: str) -> None:
    """Reject changed prompt, target, and mask inputs after preparation."""
    screen.validate_screen(frozen)
    path = screen.OUT / f"{frozen['cases'][0]['id']}.{suffix}"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError):
        screen.validate_screen(frozen)


def test_incomplete_and_wrong_manifest(frozen: dict) -> None:
    """Reject case deletion and token files missing frozen manifest identity."""
    frozen["cases"].pop()
    with pytest.raises(ValueError):
        screen.validate_screen(frozen)
    path = screen.BASE / "development-cases.tsv"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError):
        screen.prepare()


def test_wrong_frozen_token_identity(frozen: dict) -> None:
    """Preparation binds token files to the manifest, beyond the case TSV."""
    path = screen.BASE / "tokens/validation.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["windows"][0]["tokens_sha256"] = "f" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen manifest"):
        screen.prepare()


@pytest.mark.parametrize(
    "candidate,promote", [(1.02, True), (1.04, False), (1.2, False)]
)
def test_promotion_requires_both_limits(
    frozen: dict, monkeypatch: pytest.MonkeyPatch, candidate: float, promote: bool
) -> None:
    """Promotion needs improvement over Q4G64 and the comparator delta limit."""
    for arm, identity in identities().items():
        (screen.OUT / arm).mkdir()
        (screen.OUT / arm / "candidate_identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(
        screen,
        "score_arm",
        lambda arm, cases: [128 * (1.2 if arm == "q4g64" else candidate)] * len(cases),
    )
    screen.report()
    result = json.loads((screen.OUT / "report.json").read_text())
    assert result["proceed_full_gate"] is promote


def test_score_uses_stable_full_vocabulary_nll(
    frozen: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check stable NLL, then reject nonfinite real runtime logit bytes."""
    monkeypatch.setattr(screen, "VOCAB", 4)
    case = frozen["cases"][0]
    arm = screen.OUT / "q4k"
    arm.mkdir()
    (arm / "cases.jsonl").write_text(
        json.dumps(
            {
                "id": case["id"],
                "status": "complete",
                "prompt_tokens": 384,
                "target_tokens": 128,
            }
        )
        + "\n"
    )
    path = arm / f"{case['id']}.candidate-logits.f32le"
    logits = np.full((128, 4), 1000, dtype="<f4")
    logits.tofile(path)
    assert screen.score_arm("q4k", [case]) == pytest.approx([128 * np.log(4)])
    logits[0, 0] = np.nan
    logits.tofile(path)
    with pytest.raises(ValueError, match="nonfinite"):
        screen.score_arm("q4k", [case])
