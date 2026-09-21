#!/usr/bin/env python3
"""Prove Phase 1 remediation checkers reject focused temporary mutations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def first_json(path: Path) -> dict:
    match = re.search(r"```json\s*\n(.*?)```", path.read_text(), re.DOTALL)
    assert match, f"no JSON fence in {path}"
    value = json.loads(match.group(1))
    assert isinstance(value, dict)
    return value


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation anchor count for {old!r}: {text.count(old)}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def run_reject(command: list[str], label: str) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode == 0:
        raise AssertionError(f"{label}: owning checker accepted mutation")
    print(f"PASS {label}: rejected with exit {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    py = sys.executable

    cases = [
        ("TASK-02 equations/MTP", "model-semantics.md", '"mtp_semantics_status": "conditional_unverified"', '"mtp_semantics_status": "observed"', "check_model_semantics.py", ["--semantics"]),
        ("TASK-03 Diagram 6", "dataflow.md", "K_prior[(K prior)]", "K_prior_removed[(K prior)]", "check_dataflow.py", ["--dataflow"]),
        ("TASK-04 lifetime", "lifetime-and-state.md", '"primary_includes_mtp_kv": true', '"primary_includes_mtp_kv": false', "check_lifetime_and_state.py", ["--lifetime"]),
        ("TASK-06 region status", "work-and-traffic.md", '"act_region_cut_decode_language_bytes": 5245952', '"act_region_cut_decode_language_bytes": 1', "check_work_and_traffic.py", ["--work-traffic"]),
        ("TASK-07 sensitive op", "numerical-sensitivity.md", '  "sensitive_ops": [\n    "param_bf16",', '  "sensitive_ops_removed": [\n    "param_bf16",', "check_numerical_sensitivity.py", ["--numerical-sensitivity"]),
        ("TASK-09 descriptors", "runtime-format-design.md", '"logical_descriptors": {', '"logical_descriptors_removed": {', "check_runtime_format_design.py", ["--runtime-format-design"]),
        ("TASK-10 descriptors", "model-compiler-plan.md", '"logical_descriptors": {', '"logical_descriptors_removed": {', "check_model_compiler_plan.py", ["--model-compiler-plan"]),
        ("TASK-11 conditional MTP", "semantic-graph.md", '"n_mtp_mix_instances": 1', '"n_mtp_mix_instances": 0', "check_semantic_graph.py", ["--semantic-graph"]),
        ("TASK-13 state I/O", "decode-plan.md", '"stage_state_read": {', '"stage_state_read_removed": {', "check_decode_plan.py", ["--decode-plan"]),
        ("TASK-14 nine stages", "prefill-plan.md", '"n_stage_kinds_compared_with_decode": 9', '"n_stage_kinds_compared_with_decode": 8', "check_prefill_plan.py", ["--prefill-plan"]),
        ("TASK-15 view graph", "layout-strategy.md", "choice[unresolved view choice]", "choice_removed[unresolved view choice]", "check_layout_strategy.py", ["--layout-strategy"]),
        ("TASK-16 one fence", "cuda-hardware-model.md", '"n_formulae": 14', '"n_formulae": 13', "check_cuda_hardware_model.py", ["--check"], False),
        ("TASK-18 identity", "quantization-validation.md", '"measurement_identity": {', '"measurement_identity_removed": {', "check_quantization_validation.py", ["--quantization-validation"]),
        ("TASK-19 risk row", "performance-validation.md", '"methodology_risk_usefulness_label": "HYPOTHESIS"', '"methodology_risk_usefulness_label": "OBSERVED"', "check_performance_validation.py", ["--performance-validation"]),
        ("TASK-21 mechanical freeze", "clean-sheet-review.md", '"frozen_for_comparative_review": true', '"frozen_for_comparative_review": false', "check_clean_sheet_review.py", ["--review"]),
    ]

    with tempfile.TemporaryDirectory(prefix="phase1-review-") as tmp:
        tmpdir = Path(tmp)
        # TASK-05 uses the owning checker's table parser with the already
        # published live object; the one authoritative payload scan is run
        # separately by TASK-22 acceptance.
        source_analysis = ROOT / "docs/architecture/bf16-tensor-analysis.md"
        mutated_analysis = tmpdir / "bf16-tensor-analysis.md"
        shutil.copy2(source_analysis, mutated_analysis)
        replace_once(mutated_analysis, "| `pooled_all` | 27781427952 |", "| `pooled_all` | 1 |")
        spec = importlib.util.spec_from_file_location("task05_checker", ROOT / "scripts/analyze_bf16_tensors.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.check_analysis(first_json(source_analysis), mutated_analysis)
        except module.AnalysisMismatch:
            print("PASS TASK-05 numeric provenance: rejected by owning checker")
        else:
            raise AssertionError("TASK-05 numeric provenance: owning checker accepted mutation")

        for case in cases:
            label, name, old, new, checker, flag, *no_config = case
            target = tmpdir / name
            shutil.copy2(ROOT / "docs/architecture" / name, target)
            replace_once(target, old, new)
            command = [py, str(ROOT / "scripts" / checker)]
            if not no_config:
                command += ["--config", str(config)]
            command += flag + [str(target)]
            run_reject(command, label)

        # TASK-17: every keyed record, plus both corrected synchronization cases.
        source = ROOT / "docs/architecture/cuda-design-space.md"
        records = first_json(source)["mapping_records"]
        assert len(records) == 18
        for mapping_id in records:
            target = tmpdir / f"cuda-{mapping_id}.md"
            shutil.copy2(source, target)
            replace_once(target, f'"{mapping_id}": {{', f'"{mapping_id}_removed": {{')
            run_reject([py, str(ROOT / "scripts/check_cuda_design_space.py"), "--config", str(config), "--cuda-design-space", str(target)], f"TASK-17 record {mapping_id}")
        for mapping_id, old in (("map_mlp_grid_T", '"class": "sync_none",\n        "scope": "grid"'), ("map_mtp_split_norm_gemm", '"embedding_norm->concat_materialize"')):
            target = tmpdir / f"cuda-sync-{mapping_id}.md"
            shutil.copy2(source, target)
            replace_once(target, old, old + "_mutated")
            run_reject([py, str(ROOT / "scripts/check_cuda_design_space.py"), "--config", str(config), "--cuda-design-space", str(target)], f"TASK-17 sync {mapping_id}")

        # TASK-20 rendered backlog object is checked as a whole.
        arch = tmpdir / "clean-sheet-architecture.md"
        backlog = tmpdir / "experiment-backlog.md"
        shutil.copy2(ROOT / "docs/architecture/clean-sheet-architecture.md", arch)
        shutil.copy2(ROOT / "docs/architecture/experiment-backlog.md", backlog)
        replace_once(backlog, '"title": "Fill sitting SKU table"', '"title": "Mutated title"')
        run_reject([py, str(ROOT / "scripts/check_clean_sheet_architecture.py"), "--config", str(config), "--architecture", str(arch), "--experiment-backlog", str(backlog)], "TASK-20 rendered experiment field")

    print("phase1 review remediation negative mutations: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
