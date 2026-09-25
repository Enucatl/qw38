"""Focused regressions for EVAL-01 deterministic answer scoring."""

from pathlib import Path

from scripts.task018_metrics import (
    CaseMetric,
    nearest_rank,
    paired_case_bootstrap,
    score_position,
)
from scripts.task018_scoring import (
    expand_compsec_key,
    grade_c92,
    grade_l12,
    grade_retrieval,
)


def test_l12_trims_ascii_whitespace_only() -> None:
    """L12 ignores word case but preserves format and ASCII edge trimming."""
    assert grade_l12(" \tblue\n", "blue").correct
    assert grade_l12("Blue", "blue").correct
    assert grade_l12("Gatto", "gatto").correct
    assert not grade_l12("\u00a0blue\u00a0", "blue").correct
    assert not grade_l12('{"OK":true}', '{"ok":true}').correct
    assert not grade_l12("1, 2, 3", "1,2,3").correct


def test_retrieval_is_exact_after_ascii_edge_trim() -> None:
    """R does not search for an answer embedded in prose."""
    assert grade_retrieval("731204\n", "731204").correct
    assert not grade_retrieval("The code is 731204", "731204").correct


def test_c92_requires_the_last_nonempty_answer_line() -> None:
    """C92 rejects missing markers and trailing prose."""
    assert grade_c92("Reasoning\nAnswer: B\n", "GPQA Diamond", "B", 4).correct
    trailing = grade_c92("Answer: B\nExplanation", "GPQA Diamond", "B", 4)
    assert not trailing.valid
    malformed = grade_c92("The answer is B", "GPQA Diamond", "B", 4)
    assert not malformed.valid


def test_compsec_accepts_any_nonempty_key_subset_and_rejects_duplicates() -> None:
    """COMPSEC ranges represent alternative acceptable locations."""
    assert expand_compsec_key("3,13-15") == {3, 13, 14, 15}
    assert grade_c92("Answer: 20", "COMPSEC", "17-20").correct
    assert grade_c92("Answer: 18, 20", "COMPSEC", "17-20").correct
    assert not grade_c92("Answer: 18,18", "COMPSEC", "17-20").valid
    assert not grade_c92("Answer: 21", "COMPSEC", "17-20").correct


def test_c92_domain_extractors_enforce_value_formats() -> None:
    """Multiple choice and AIME answers use exact family-specific formats."""
    assert grade_c92("Answer: J", "SuperGPQA", "J", 10).correct
    assert not grade_c92("Answer: K", "SuperGPQA", "A", 10).valid
    assert grade_c92("Answer: 007", "AIME2025", "7").correct
    assert not grade_c92("Answer: 1000", "AIME2025", "0").valid


def test_vendor_input_identities_and_inventory() -> None:
    """The local DS4 source files still match the selected EVAL-01 identities."""
    import hashlib
    import json

    root = Path(__file__).resolve().parents[2] / "ds4"
    prompts = root / "gguf-tools/quality-testing/prompts.jsonl"
    questions = root / "ds4_eval.c"
    assert hashlib.sha256(prompts.read_bytes()).hexdigest() == (
        "007757e6c4b340c209aba8a7e159024ce43b0edb237547cdbc3ca51a4853999a"
    )
    assert hashlib.sha256(questions.read_bytes()).hexdigest() == (
        "19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23"
    )
    rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    assert [row["id"] for row in rows[:100]] == [f"case_{i:03d}" for i in range(100)]


def test_full_distribution_math_uses_stable_softmax_and_tie_order() -> None:
    """Equal logits have log(V) NLL, zero KL, and ID-stable top-k."""
    import math

    import numpy as np

    logits = np.zeros(20, dtype=np.float32)
    result = score_position(logits, logits, 3, require_qw38_vocabulary=False)
    assert abs(result.nll_reference - math.log(20)) < 1e-6
    assert abs(result.nll_candidate - math.log(20)) < 1e-6
    assert abs(result.kl_reference_to_candidate) < 1e-12
    assert result.top1_agreement
    assert result.top20_overlap_fraction == 1.0


def test_distribution_math_rejects_nonfinite_and_wrong_qw38_vocab() -> None:
    """Non-finite and undersized production readouts are invalid evidence."""
    import numpy as np
    import pytest

    with pytest.raises(ValueError, match="finite"):
        score_position(
            np.array([0.0, np.nan]),
            np.array([0.0, 1.0]),
            0,
            require_qw38_vocabulary=False,
        )
    with pytest.raises(ValueError, match="vocabulary size"):
        score_position(np.zeros(20), np.zeros(20), 0)


def test_paired_bootstrap_is_deterministic_and_case_weighted() -> None:
    """SHA-256 resampling pairs model outcomes and honors separate families."""
    import pytest

    rows = [
        CaseMetric("a", "P100", 2.0, 2.2, 2, 1, 1),
        CaseMetric("b", "P100", 4.0, 4.5, 4, 0, 0),
        CaseMetric("c", "C92-GPQA", 1.0, 1.0, 1, 1, 0),
    ]
    left = paired_case_bootstrap(rows, metric_group="test", replicates=100)
    right = paired_case_bootstrap(rows, metric_group="test", replicates=100)
    assert left == right
    assert 0 <= left["nll_delta_low_95"] <= left["nll_delta_high_95"]
    assert 0 <= left["accuracy_loss_low_95"] <= left["accuracy_loss_high_95"]
    assert nearest_rank([0.0, 1.0, 2.0, 3.0], 0.5) == 1.0
    with pytest.raises(ValueError, match="duplicate case"):
        paired_case_bootstrap(rows + [rows[0]], metric_group="test", replicates=10)


def test_report_bootstrap_accuracy_uses_the_paired_sampler() -> None:
    """The report's C92 aggregate invokes its accuracy adapter successfully."""
    from scripts.task018_pair_report import bootstrap_accuracy

    rows = [CaseMetric("a", "C92", 1.0, 1.0, 1, 1, 0)]
    result = bootstrap_accuracy(rows, "C92/all")
    assert result["accuracy_loss_low_95"] == 1.0
    assert result["accuracy_loss_high_95"] == 1.0


def test_pair_report_rejects_alignment_mask_hash_and_replay_failures(
    tmp_path: Path,
) -> None:
    """Focused negative checks keep malformed paired evidence out of scoring."""
    import pytest

    from scripts.task018_pair_report import (
        require_fresh_llama_generation,
        require_hash,
        require_mask,
        require_replay_match,
        require_target_alignment,
        write_unscored_result,
    )

    with pytest.raises(ValueError, match="teacher probability targets"):
        require_target_alignment(
            [4, 5], [{"target_id": 4}, {"target_id": 9}], "alignment"
        )
    with pytest.raises(ValueError, match="loss mask count"):
        require_mask(b"\x01\x00", 2, "mask")
    evidence = tmp_path / "identity.bin"
    evidence.write_bytes(b"original")
    with pytest.raises(ValueError, match="hash differs"):
        require_hash(evidence, "0" * 64, "fixture")
    with pytest.raises(ValueError, match="fresh llama request replay"):
        require_replay_match([1, 2], [1, 3], "replay")
    with pytest.raises(ValueError, match="reused without source-run provenance"):
        require_fresh_llama_generation({"reused_generation_from": "failed-run"})
    require_fresh_llama_generation({})
    invalid_dir = tmp_path / "invalid"
    write_unscored_result(
        invalid_dir, "INVALID", "ValueError: intentionally malformed evidence"
    )
    import json

    invalid = json.loads((invalid_dir / "result.json").read_text())
    assert invalid["status"] == "INVALID"
    assert invalid["scoring_performed"] is False
    assert "malformed evidence" in invalid["reason"]
    write_unscored_result(invalid_dir, "INCONCLUSIVE", "incomplete arm coverage")
    assert (
        json.loads((invalid_dir / "result.json").read_text())["status"]
        == "INCONCLUSIVE"
    )


def test_policy_rebind_preserves_fixture_identity(tmp_path: Path) -> None:
    """Rebinding requires the exact old fixture and current policy bytes."""
    import hashlib
    import json

    import pytest

    from scripts.task018_pair_report import validate_policy_rebind

    policy = tmp_path / "policy.md"
    policy.write_text("reconciled policy\n")
    old = "0" * 64
    manifest = {
        "policy_path": str(policy),
        "policy_sha256": old,
        "files": {"prompts.jsonl": {"sha256": "1" * 64}},
        "reference_capture": {"source_refs_jsonl": {"sha256": "2" * 64}},
        "teacher_probability_capture": {"sha256": "3" * 64},
        "scoring_implementation": {"sha256": "4" * 64},
        "metrics_implementation": {"sha256": "5" * 64},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rebind = {
        "schema": "qw38-eval-policy-rebind-v1",
        "fixture_manifest_sha256": hashlib.sha256(
            (tmp_path / "manifest.json").read_bytes()
        ).hexdigest(),
        "previous_policy_sha256": old,
        "effective_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "prompts_sha256": "1" * 64,
        "teacher_refs_sha256": "2" * 64,
        "teacher_probabilities_sha256": "3" * 64,
        "scoring_sha256": "4" * 64,
        "metrics_sha256": "5" * 64,
    }
    path = tmp_path / "rebind.json"
    path.write_text(json.dumps(rebind))
    assert validate_policy_rebind(tmp_path, manifest, path) == (
        rebind["effective_policy_sha256"],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    rebind["teacher_refs_sha256"] = "9" * 64
    path.write_text(json.dumps(rebind))
    with pytest.raises(ValueError, match="policy rebind does not match"):
        validate_policy_rebind(tmp_path, manifest, path)
    with pytest.raises(ValueError, match="policy hash differs"):
        validate_policy_rebind(tmp_path, manifest, None)


def test_fixed_answer_targets_and_basic_correctness_gate(tmp_path: Path) -> None:
    """L12 reads the run-local target and wrong fixed answers block acceptance."""
    from scripts.task018_pair_report import (
        fixed_answer_target_path,
        summarize_fixed_answers,
    )

    run = tmp_path / "v0-run"
    fixtures = tmp_path / "fixtures"
    l12 = {"id": "L01", "family": "L12"}
    retrieval = {
        "id": "R-512-s0-d0.1",
        "family": "R",
        "details": {"target_token_file": "tokens/r.target.u32le"},
    }
    assert (
        fixed_answer_target_path(run, fixtures, l12) == run / "targets/L01.target.u32le"
    )
    assert (
        fixed_answer_target_path(run, fixtures, retrieval)
        == fixtures / "tokens/r.target.u32le"
    )

    rows = [
        {
            "id": "L01",
            "llama_answer": {"correct": True},
            "v0_answer": {"correct": False},
            "llama_stop": "eos",
            "v0_stop": "cap",
        }
    ]
    summary = summarize_fixed_answers(rows)
    assert summary["status"] == "FAIL"
    assert summary["arms"]["v0"]["incorrect_ids"] == ["L01"]
    assert summary["arms"]["v0"]["cap_exhausted_ids"] == ["L01"]
    rows[0]["v0_answer"]["correct"] = True
    assert summarize_fixed_answers(rows)["status"] == "PASS"
