"""Host tests for OPT-068 scoped O2/O3/FMA Q4 prompt-MMQ codegen."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt068_scoped_codegen_contract.json"
ITERATION = ROOT / "pins/opt068_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt068_scoped_codegen.json"
REPORT = ROOT / "evidence/optimization/opt068-scoped-codegen/REPORT.md"
MAKEFILE = ROOT / "Makefile"
NATIVE = ROOT / "cuda/opt068_codegen_test.cu"
DRIVER = ROOT / "cuda/opt068_codegen_driver.cpp"
FAMILY_TU = ROOT / "cuda/q4_prompt_mmq.cu"
MMV = ROOT / "cuda/quant_mmv.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_native_target() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-068")
    fixture = _json(FIXTURE)
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert DRIVER.is_file()
    assert FAMILY_TU.is_file()
    assert contract["task"] == "OPT-068"
    assert contract["claims_throughput"] is False
    assert contract["family"] == "q4_prompt_mmq"
    assert contract["variants"] == [
        "o2_fmad_false",
        "o3_fmad_false",
        "o3_fmad_true",
    ]
    assert iteration["task"] == "OPT-068"
    assert iteration["target"] == "build/qw38-cuda-opt068-codegen-test"
    assert iteration["diagnostics_make_target"] == "cuda-opt068-diagnostics"
    assert fixture["family"] == "q4_prompt_mmq"
    assert fixture["opt061_rank"] == "prompt-ffn"
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt068-codegen-test" in makefile
    assert "cuda-opt068-diagnostics" in makefile
    assert "OPT068_O2_DIR" in makefile
    assert "OPT068_O3_DIR" in makefile
    assert "OPT068_O3FMA_DIR" in makefile
    assert "o2_fmad_false" in makefile
    assert "o3_fmad_false" in makefile
    assert "o3_fmad_true" in makefile


def test_source_extracts_family_and_keeps_strict_flags() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    mmv = MMV.read_text(encoding="utf-8")
    family = FAMILY_TU.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    assert "NVCCFLAGS := -std=c++17 -O2 -arch=sm_120" in makefile
    assert "--fmad=false" in makefile
    assert "-ffp-contract=off" in makefile
    assert "use_fast_math" not in makefile
    assert "maxrregcount" not in makefile
    assert "OPT068_O3FMA_NVCCFLAGS" in makefile
    assert "--fmad=true" in makefile
    nvcc_line = next(
        line for line in makefile.splitlines() if line.startswith("NVCCFLAGS :=")
    )
    assert "-O2" in nvcc_line
    assert "--fmad=false" in nvcc_line
    assert "--fmad=true" not in nvcc_line
    assert '#include "quant_mmq_mma.cuh"' not in mmv
    assert '#include "quant_mmq_mma.cuh"' in family
    assert "quant_mmq_mma.cuh" not in scheduler
    assert "q4_prompt_mmq" in native
    assert "o2_fmad_false" in driver
    assert "o3_fmad_true" in driver
    assert "opt061" in native.casefold() or "OPT-061" in native


def test_iteration_loop_products_and_no_oracles() -> None:
    iteration = load_contract("OPT-068")
    assert iteration["claims_throughput"] is False
    assert iteration["modes"]["feedback"]["tier_sequence"] == [
        "smoke",
        "correctness",
    ]
    assert iteration["modes"]["acceptance"]["tier_sequence"] == [
        "smoke",
        "correctness",
        "screen",
    ]
    assert loop_product(iteration["workloads"]["smoke"]) == 3
    assert loop_product(iteration["workloads"]["correctness"]) == 18
    assert loop_product(iteration["workloads"]["screen"]) == 12
    assert loop_product(iteration["workloads"]["acceptance"]) == 12
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "q4 prompt mmq" in proof
    assert "flag stamps" in proof
    assert "use_fast_math" in proof
    assert "no tok/s" in proof
    assert iteration["modes"]["release"]["historical_oracles"] is False


def test_makefile_stamp_invalidates_on_flag_change(tmp_path: Path) -> None:
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        """
BUILD := build
O2 := $(BUILD)/cuda/opt068/o2_fmad_false
O3 := $(BUILD)/cuda/opt068/o3_fmad_false
STAMP := $(O2)/nvccflags.stamp
STAMP3 := $(O3)/nvccflags.stamp
FLAGS ?= -O2 --fmad=false
FLAGS3 ?= -O3 --fmad=false
CC ?= cc

$(BUILD) $(O2) $(O3):
	mkdir -p $@

.PHONY: FORCE all
FORCE:

$(STAMP): FORCE | $(O2)
	@printf '%s\\n' '$(FLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(STAMP3): FORCE | $(O3)
	@printf '%s\\n' '$(FLAGS3)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(O2)/family.o: family.c $(STAMP) | $(O2)
	$(CC) -c family.c -o $@

all: $(O2)/family.o $(STAMP3)
"""
    )
    (tmp_path / "family.c").write_text("int family(void) { return 68; }\n")

    def make(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["make", "-C", str(tmp_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    first = make("all")
    assert first.returncode == 0, first.stdout + first.stderr
    obj = tmp_path / "build/cuda/opt068/o2_fmad_false/family.o"
    stamp = tmp_path / "build/cuda/opt068/o2_fmad_false/nvccflags.stamp"
    stamp3 = tmp_path / "build/cuda/opt068/o3_fmad_false/nvccflags.stamp"
    assert obj.is_file()
    first_mtime = obj.stat().st_mtime_ns
    first_stamp = stamp.read_text()
    assert stamp.read_text() != stamp3.read_text()
    second = make("all")
    assert second.returncode == 0, second.stdout + second.stderr
    assert obj.stat().st_mtime_ns == first_mtime
    third = make("all", "FLAGS=-O3 --fmad=false")
    assert third.returncode == 0, third.stdout + third.stderr
    assert obj.stat().st_mtime_ns != first_mtime
    assert stamp.read_text() != first_stamp
    assert "fmad=false" in stamp3.read_text()


def test_report_records_family_and_flag_isolation() -> None:
    text = REPORT.read_text(encoding="utf-8").casefold()
    fixture = _json(FIXTURE)
    assert "opt-068" in text
    assert "q4 prompt mmq" in text or "q4_prompt_mmq" in text
    assert "opt-061" in text
    assert "-o2" in text
    assert "-o3" in text
    assert "fmad" in text
    assert "retain" in text or "keep" in text
    assert fixture["selected_flags"] == "-O2 --fmad=false"
    assert fixture["strict_host_unchanged"] is True
    assert fixture["use_fast_math"] is False
    assert fixture["claims_throughput"] is False
    assert fixture["keep"] is False
    assert fixture["installed"] is False
    assert fixture["winner"] == "o2_fmad_false"
    assert fixture["o2_complete_ffn_ms"] == 8.33649063
    assert fixture["o3_complete_ffn_ms"] == 8.34392548
    assert fixture["fma_complete_ffn_ms"] == 8.34093857
    assert "8.336" in text
    assert "8.343" in text
    assert "rejected" in text or "retain" in text
    for key in _json(CONTRACT)["required_fixture_keys"]:
        assert key in fixture, key
