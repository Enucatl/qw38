#!/usr/bin/env python3
"""Check the model-independent CUDA hardware-model catalog and markdown.

Builds the locked catalog object (no ``config.json``, no GPU query) and either
prints it as JSON or checks that ``docs/architecture/cuda-hardware-model.md``
matches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUTHORITY = "docs/architecture/plan.md"
DELIVERABLE = "docs/architecture/cuda-hardware-model.md"
SKU_POLICY = "parameterized_unknown_until_measured_table"

CANONICAL_SKU_SENTENCE = (
    "Target-GPU-specific limits and instruction availability remain UNKNOWN "
    "until a future measured SKU table. This document supplies the symbols "
    "and evaluation criteria TASK-17 will instantiate; it does not record "
    "sitting-device numbers and does not claim a winning fusion policy."
)
CANONICAL_FUSION_SENTENCE = (
    "Fusion versus occupancy algebra is an evaluation criterion, not a "
    "selected mapping."
)
CANONICAL_WINNER_SENTENCE = "This document selects no CUDA mapping winner."

HEADINGS: tuple[str, ...] = (
    "Authority",
    "SKU parameterization",
    "Execution hierarchy",
    "Occupancy and latency hiding",
    "Memory hierarchy",
    "Access patterns",
    "Resource limiters",
    "Synchronization",
    "Instruction pipelines",
    "Fusion versus occupancy",
    "Evaluation criteria",
    "Deferred SKU table",
    "Machine-checkable catalog",
)

CONCEPT_IDS: tuple[str, ...] = (
    "thread",
    "warp",
    "warp_size",
    "simt_divergence",
    "cta",
    "block",
    "grid",
    "sm",
    "scheduler",
    "stall",
    "occupancy",
    "theoretical_occupancy",
    "latency_hiding",
    "wave_quantization",
    "thread_block_cluster",
    "registers",
    "shared_memory",
    "shared_memory_banks",
    "l1",
    "l2",
    "hbm",
    "host_pinned",
    "local_memory",
    "coalescing",
    "alignment",
    "bank_conflicts",
    "capacity",
    "bandwidth",
    "latency_resource",
    "registers_per_thread",
    "shared_mem_per_cta",
    "threads_per_cta",
    "ctas_per_sm",
    "warp_slots",
    "barrier_slots",
    "register_spilling",
    "littles_law",
    "syncthreads",
    "warp_sync",
    "memory_fence",
    "stream",
    "event",
    "cooperative_groups",
    "async_copy_pipeline",
    "fma",
    "ffma",
    "load_store",
    "tensor_core_mma",
    "async_gmem_to_smem",
    "tma",
    "occupancy_min",
    "fusion_footprint",
    "arithmetic_intensity",
    "roofline",
)

FORMULA_IDS: tuple[str, ...] = (
    "occupancy_min",
    "limiter_reg",
    "limiter_smem",
    "limiter_threads",
    "limiter_cta",
    "resident_cta",
    "warps_per_cta",
    "occupancy_warps",
    "littles_law",
    "wave_quant",
    "fusion_footprint",
    "fusion_smem",
    "arithmetic_intensity",
    "roofline",
)

FORMULA_SUBSTRINGS: tuple[str, ...] = (
    r"O = \min(O_\text{reg}, O_\text{smem}, O_\text{threads}, O_\text{cta})",
    r"B_\text{reg} = \lfloor S_\text{reg} / R_\text{cta} \rfloor",
    r"B_\text{smem} = \lfloor C_\text{smem} / C_\text{alloc} \rfloor",
    r"B_\text{threads} = \lfloor T_\max / T_\text{cta} \rfloor",
    r"B_\text{cta} = B_\max",
    r"B_\text{SM} = \min(B_\text{reg}, B_\text{smem}, B_\text{threads}, B_\text{cta})",
    r"W_\text{cta} = \lceil T_\text{cta} / N_w \rceil",
    r"O = W_\text{active} / W_\max",
    r"W_\text{need} = N_\text{sched} \cdot L_\text{issue}",
    r"N_\text{waves} = \lceil N_\text{grid} / (N_\text{SM} \cdot B_\text{SM}) \rceil",
    r"R_f \ge \max(R_1, R_2)",
    r"C_f \ge \max(C_1, C_2)",
    r"I = F / B",
    r"\Pi \le \min(\Pi_\text{peak}, I \cdot \Beta)",
)

SKU_UNKNOWN_SYMBOLS: tuple[str, ...] = (
    "N_SM",
    "W_max",
    "S_reg",
    "C_smem",
    "T_max",
    "B_max",
    "N_bar",
    "N_sched",
    "G_reg",
    "G_smem",
    "Beta_HBM",
    "Pi_FMA",
    "Pi_TC",
    "L_issue",
    "async_copy_cap",
    "mma_shapes",
    "cluster_cap",
)

FORBIDDEN_TOKENS: tuple[str, ...] = (
    "qwen",
    "quartz",
    "llama.cpp",
    "ggml",
    "gguf",
    "opt-",
    ".cu",
    ".cuh",
    "TBD",
    "TODO",
    "???",
    "MEASURED",
    "HYPOTHESIS",
)

FORBIDDEN_NAME_TOKENS: tuple[str, ...] = (
    "qwen",
    "quartz",
    "llama.cpp",
    "ggml",
    "gguf",
    "opt-",
)
FORBIDDEN_PATH_TOKENS: tuple[str, ...] = (".cu", ".cuh")
PLACEHOLDER_TOKENS: tuple[str, ...] = ("TBD", "TODO", "???")
FORBIDDEN_LABEL_TOKENS: tuple[str, ...] = ("MEASURED", "HYPOTHESIS")

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "deliverable",
    "sku_policy",
    "warp_size",
    "n_bank",
    "n_headings",
    "headings",
    "n_concepts",
    "concept_ids",
    "n_formulae",
    "formula_ids",
    "formula_substrings",
    "sku_unknown_symbols",
    "forbidden_tokens",
    "canonical_sku_sentence",
    "canonical_fusion_sentence",
    "canonical_winner_sentence",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (?!#)(.+)$", re.MULTILINE)


class HardwareModelMismatch(Exception):
    """Raised when the hardware-model markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_catalog(catalog: dict) -> str:
    """Pretty-print the locked hardware-model catalog.

    Args:
        catalog: Object produced by :func:`build_catalog`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(catalog, indent=2) + "\n"


def first_json_fence(text: str) -> tuple[dict, re.Match[str]]:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown document containing a `` ```json `` fence.

    Returns:
        Decoded JSON object and the regex match covering the fence.

    Raises:
        AssertionError: If no fence exists or the payload is not an object.
        json.JSONDecodeError: If the fence is not valid JSON.
    """

    match = JSON_FENCE_RE.search(text)
    assert match is not None, "no fenced json block found"
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict), "fenced json block is not an object"
    return payload, match


def build_catalog() -> dict:
    """Return the locked CUDA hardware-model catalog object.

    Returns:
        Catalog dict with schema keys in locked order.

    Raises:
        AssertionError: If an internal identity fails.
    """

    catalog = {
        "authority": AUTHORITY,
        "deliverable": DELIVERABLE,
        "sku_policy": SKU_POLICY,
        "warp_size": 32,
        "n_bank": 32,
        "n_headings": 13,
        "headings": list(HEADINGS),
        "n_concepts": 54,
        "concept_ids": list(CONCEPT_IDS),
        "n_formulae": 14,
        "formula_ids": list(FORMULA_IDS),
        "formula_substrings": list(FORMULA_SUBSTRINGS),
        "sku_unknown_symbols": list(SKU_UNKNOWN_SYMBOLS),
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "canonical_sku_sentence": CANONICAL_SKU_SENTENCE,
        "canonical_fusion_sentence": CANONICAL_FUSION_SENTENCE,
        "canonical_winner_sentence": CANONICAL_WINNER_SENTENCE,
    }
    _assert_catalog(catalog)
    return catalog


def _assert_catalog(catalog: dict) -> None:
    """Assert dossier identities on a live catalog object.

    Args:
        catalog: Object produced by :func:`build_catalog`.

    Raises:
        AssertionError: If a locked identity does not hold.
    """

    assert list(catalog) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(catalog)} vs {list(SCHEMA_KEYS)}"
    )
    assert catalog["warp_size"] == 32
    assert catalog["n_bank"] == 32
    assert catalog["n_headings"] == 13 == len(HEADINGS)
    assert catalog["n_headings"] == len(catalog["headings"])
    assert catalog["n_concepts"] == 54 == len(CONCEPT_IDS)
    assert catalog["n_concepts"] == len(catalog["concept_ids"])
    assert catalog["n_formulae"] == 14 == len(FORMULA_IDS) == len(
        FORMULA_SUBSTRINGS
    )
    assert catalog["n_formulae"] == len(catalog["formula_ids"])
    assert catalog["n_formulae"] == len(catalog["formula_substrings"])
    assert len(set(CONCEPT_IDS)) == len(CONCEPT_IDS)
    assert catalog["concept_ids"] == list(CONCEPT_IDS)
    assert catalog["sku_policy"] == SKU_POLICY
    assert catalog["authority"] == AUTHORITY
    assert catalog["deliverable"] == DELIVERABLE
    assert catalog["headings"] == list(HEADINGS)
    assert catalog["formula_ids"] == list(FORMULA_IDS)
    assert catalog["formula_substrings"] == list(FORMULA_SUBSTRINGS)
    assert catalog["sku_unknown_symbols"] == list(SKU_UNKNOWN_SYMBOLS)
    assert catalog["forbidden_tokens"] == list(FORBIDDEN_TOKENS)
    assert catalog["canonical_sku_sentence"] == CANONICAL_SKU_SENTENCE
    assert catalog["canonical_fusion_sentence"] == CANONICAL_FUSION_SENTENCE
    assert catalog["canonical_winner_sentence"] == CANONICAL_WINNER_SENTENCE


def diff_catalog(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live catalog object.

    Args:
        live: Fresh script object.
        documented: Object parsed from a markdown JSON fence.

    Returns:
        Human-readable difference strings; empty when the documents match.
    """

    diffs: list[str] = []
    live_keys = list(live)
    doc_keys = list(documented)
    if doc_keys != live_keys:
        extra = [key for key in doc_keys if key not in live]
        missing = [key for key in live_keys if key not in documented]
        if extra:
            diffs.append(f"documented extra keys: {extra}")
        if missing:
            diffs.append(f"documented missing keys: {missing}")
        if not extra and not missing and doc_keys != live_keys:
            diffs.append(f"documented key order {doc_keys} != live {live_keys}")
    if documented != live:
        for key in SCHEMA_KEYS:
            if key not in documented or key not in live:
                continue
            if documented[key] != live[key]:
                diffs.append(
                    f"{key}: documented {documented[key]!r} != live {live[key]!r}"
                )
        if not any(key in documented and key in live for key in SCHEMA_KEYS):
            diffs.append("documented JSON is not deep-equal to the live object")
    return diffs


def _line_number(text: str, index: int) -> int:
    """Return the 1-based line number of ``index`` in ``text``."""

    return text.count("\n", 0, index) + 1


def _scan_forbidden(search_text: str, differences: list[str]) -> None:
    """Append forbidden-token mismatches for ``search_text``.

    Args:
        search_text: Markdown with the first JSON fence removed.
        differences: Accumulator of human-readable mismatch strings.
    """

    for token in PLACEHOLDER_TOKENS:
        for match in re.finditer(re.escape(token), search_text):
            line = _line_number(search_text, match.start())
            differences.append(f"line {line}: forbidden token {token!r}")
    for token in FORBIDDEN_NAME_TOKENS:
        for match in re.finditer(re.escape(token), search_text, flags=re.IGNORECASE):
            line = _line_number(search_text, match.start())
            differences.append(
                f"line {line}: forbidden token {match.group(0)!r}"
            )
    for token in FORBIDDEN_PATH_TOKENS:
        for match in re.finditer(re.escape(token), search_text):
            line = _line_number(search_text, match.start())
            differences.append(f"line {line}: forbidden path token {token!r}")
    for token in FORBIDDEN_LABEL_TOKENS:
        for match in re.finditer(re.escape(token), search_text):
            line = _line_number(search_text, match.start())
            differences.append(f"line {line}: forbidden token {token!r}")


def check_hardware_model(live: dict, model_path: Path) -> None:
    """Check a hardware-model markdown file against a live catalog object.

    Args:
        live: Catalog object from :func:`build_catalog`.
        model_path: Path to the hardware-model markdown.

    Raises:
        HardwareModelMismatch: On heading, JSON, token, or substring mismatches.
        OSError: If the file cannot be read.
    """

    text = model_path.read_text(encoding="utf-8")
    differences: list[str] = []

    headings = HEADING_RE.findall(text)
    if headings != list(HEADINGS):
        differences.append(
            "heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(HEADINGS)}"
        )

    fence_match: re.Match[str] | None = None
    try:
        documented, fence_match = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_catalog(live, documented))

    if fence_match is not None:
        search_text = text[: fence_match.start()] + text[fence_match.end() :]
    else:
        search_text = text

    for concept_id in CONCEPT_IDS:
        token = f"`{concept_id}`"
        if token not in search_text:
            differences.append(
                f"concept id {concept_id!r} missing as backtick-wrapped token"
            )

    for index in range(1, 15):
        tag = f"(F{index})"
        if tag not in search_text:
            differences.append(f"formula tag {tag} missing")

    for substring in FORMULA_SUBSTRINGS:
        if substring not in search_text:
            differences.append(f"formula substring missing: {substring}")

    for symbol in SKU_UNKNOWN_SYMBOLS:
        if symbol not in search_text:
            differences.append(f"sku unknown symbol {symbol!r} missing")

    if "N_w = 32" not in search_text and "N_w=32" not in search_text:
        differences.append("warp_size identity N_w = 32 or N_w=32 missing")

    if CANONICAL_SKU_SENTENCE not in search_text:
        differences.append("canonical SKU sentence not present verbatim")
    if CANONICAL_FUSION_SENTENCE not in search_text:
        differences.append("canonical fusion sentence not present verbatim")
    if CANONICAL_WINNER_SENTENCE not in search_text:
        differences.append("canonical winner sentence not present verbatim")

    _scan_forbidden(search_text, differences)

    if differences:
        raise HardwareModelMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the hardware-model checker.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Print the locked CUDA hardware-model catalog or check "
            "docs/architecture/cuda-hardware-model.md against it."
        )
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the catalog JSON to stdout.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        metavar="PATH",
        help="Markdown path whose catalog fence and tokens must match.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CUDA hardware-model checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch/missing file.
    """

    args = _parse_args(argv)
    try:
        catalog = build_catalog()
        if args.check is not None:
            if not args.check.is_file():
                print(
                    f"hardware model file not found: {args.check}",
                    file=sys.stderr,
                )
                return 1
            check_hardware_model(catalog, args.check)
    except HardwareModelMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"assert failed: invalid catalog JSON: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"hardware model file not found: {args.check}", file=sys.stderr)
        return 1

    if args.json or args.check is None:
        sys.stdout.write(dumps_catalog(catalog))
    return 0


if __name__ == "__main__":
    sys.exit(main())
