#!/usr/bin/env python3
"""Check Qwen3.8-27B logical dataflow graph against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the graph-summary object or checks that
``docs/architecture/dataflow.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"

CANONICAL_SENTENCE = (
    "Logical values in this document are graph nodes. They do not imply "
    "physical allocation, materialization, buffer reuse, or kernel fusion."
)

CATALOG_IDS: tuple[str, ...] = (
    "token_id",
    "e",
    "h",
    "h_tilde",
    "h_mid",
    "h_post",
    "h_64",
    "h_final",
    "logits_0",
    "u_q",
    "q_prime",
    "g",
    "k_raw",
    "v_full",
    "q_n",
    "k_n",
    "q_rope",
    "k_rope",
    "attn",
    "y_gate",
    "mix_full",
    "K_state",
    "V_state",
    "qkv",
    "z",
    "a",
    "b",
    "c_tilde",
    "c",
    "q_lin",
    "k_lin",
    "v_lin",
    "q_hat",
    "k_hat",
    "alpha",
    "beta",
    "S",
    "o",
    "u_gdn",
    "mix_lin",
    "C_state",
    "g_mlp",
    "up",
    "swiglu",
    "mlp_out",
    "e_next",
    "e_next_n",
    "h64_n",
    "mtp_cat",
    "mtp_u",
    "h_mtp",
    "logits_1",
)

HIGH_FANOUT_IDS: tuple[str, ...] = (
    "h",
    "h_tilde",
    "h_mid",
    "h_post",
    "h_64",
    "k_rope",
    "v_full",
    "qkv",
    "S",
)

LIVE_ACROSS_IDS: tuple[str, ...] = ("g", "z")
INTRA_EQUATION_REUSE_IDS: tuple[str, ...] = ("k_hat",)
SHARED_WEIGHT_IDS: tuple[str, ...] = ("E", "W_lm")
STATE_IDS: tuple[str, ...] = ("K_state", "V_state", "C_state", "S")
REGIONS: tuple[str, ...] = (
    "embed",
    "decoder_stack",
    "residual_layer",
    "full_attn",
    "linear_attn",
    "mlp",
    "persistent_state",
    "primary_logits",
    "mtp",
    "vision_interface",
)

LOCKED_FULL_ATTENTION_INDICES: tuple[int, ...] = (
    3,
    7,
    11,
    15,
    19,
    23,
    27,
    31,
    35,
    39,
    43,
    47,
    51,
    55,
    59,
    63,
)

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "hidden_size",
    "n_decoder_layers",
    "n_linear_layers",
    "n_full_layers",
    "n_mtp_blocks",
    "full_attention_indices",
    "n_diagrams",
    "n_catalog_nodes",
    "catalog_ids",
    "high_fanout_ids",
    "live_across_ids",
    "intra_equation_reuse_ids",
    "shared_weight_ids",
    "state_ids",
    "regions",
    "canonical_sentence",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Logical versus physical",
    "Graph vocabulary",
    "Top-level token map",
    "Residual decoder layer",
    "Full attention",
    "Linear attention",
    "MLP",
    "Persistent state transitions",
    "Primary logits and MTP",
    "Logical intermediates",
    "Sharing and reuse",
    "Deferred vision",
    "Machine-checkable graph summary",
)

DIAGRAM_REQUIRED_IDS: tuple[tuple[str, ...], ...] = (
    (
        "token_id",
        "e",
        "vision_if",
        "h",
        "h_64",
        "h_final",
        "logits_0",
        "e_next",
        "logits_1",
        "K_state",
        "V_state",
        "C_state",
        "S",
    ),
    ("h", "h_tilde", "mix_lin", "mix_full", "h_mid", "h_post", "mlp_out"),
    (
        "h_tilde",
        "u_q",
        "q_prime",
        "g",
        "k_raw",
        "v_full",
        "q_n",
        "k_n",
        "q_rope",
        "k_rope",
        "attn",
        "y_gate",
        "mix_full",
        "K_state",
        "V_state",
    ),
    (
        "h_tilde",
        "qkv",
        "z",
        "a",
        "b",
        "c_tilde",
        "c",
        "q_lin",
        "k_lin",
        "v_lin",
        "q_hat",
        "k_hat",
        "alpha",
        "beta",
        "S",
        "o",
        "u_gdn",
        "mix_lin",
        "C_state",
    ),
    ("h_post", "g_mlp", "up", "swiglu", "mlp_out"),
    ("K_prior", "V_prior", "C_prior", "S_prior", "attention", "fir", "recurrence", "K_next", "V_next", "C_next", "S_next", "k_rope", "v_full", "qkv", "alpha", "beta", "k_hat", "v_lin", "o"),
    (
        "h_64",
        "h_final",
        "logits_0",
        "e_next",
        "e_next_n",
        "h64_n",
        "mtp_cat",
        "mtp_u",
        "h_mtp",
        "logits_1",
        "E",
        "W_lm",
    ),
    (
        "h",
        "h_tilde",
        "h_mid",
        "h_post",
        "h_64",
        "k_rope",
        "v_full",
        "qkv",
        "S",
        "g",
        "z",
        "E",
        "W_lm",
    ),
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class DataflowMismatch(Exception):
    """Raised when the dataflow markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_summary(summary: dict) -> str:
    """Pretty-print the graph-summary object.

    Args:
        summary: Object produced by :func:`instantiate_graph_summary`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(summary, indent=2) + "\n"


def first_json_fence(text: str) -> dict:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown document containing a `` ```json `` fence.

    Returns:
        Decoded JSON object.

    Raises:
        AssertionError: If no fence exists or the payload is not an object.
    """

    match = JSON_FENCE_RE.search(text)
    assert match is not None, "no fenced json block found"
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict), "fenced json block is not an object"
    return payload


def _require_int(mapping: dict, key: str) -> int:
    """Return ``mapping[key]`` as an ``int``.

    Args:
        mapping: Config object.
        key: Required field name.

    Returns:
        Integer field value.

    Raises:
        AssertionError: If the field is missing or not an integer value.
    """

    assert key in mapping, f"missing text_config field: {key}"
    value = mapping[key]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"text_config.{key} is not an int: {value!r}"
    )
    return value


def instantiate_graph_summary(config: dict) -> dict:
    """Compute the TASK-03 graph-summary object from ``config``.

    Args:
        config: Parsed Transformers ``config.json`` object.

    Returns:
        Ordered summary dict matching the dossier JSON schema.

    Raises:
        AssertionError: If required fields are missing or identities fail.
    """

    assert isinstance(config, dict), "config.json root is not an object"
    assert "text_config" in config, "config.json has no text_config"
    text = config["text_config"]
    assert isinstance(text, dict), "text_config is not an object"

    hidden_size = _require_int(text, "hidden_size")
    n_decoder_layers = _require_int(text, "num_hidden_layers")
    full_attention_interval = _require_int(text, "full_attention_interval")
    n_mtp_blocks = _require_int(text, "mtp_num_hidden_layers")

    assert "layer_types" in text, "missing layer_types"
    layer_types = text["layer_types"]
    assert isinstance(layer_types, list), "layer_types is not a list"
    assert len(layer_types) == n_decoder_layers, (
        f"layer_types length {len(layer_types)} != num_hidden_layers "
        f"{n_decoder_layers}"
    )
    full_from_types = [
        index for index, name in enumerate(layer_types) if name == "full_attention"
    ]
    linear_from_types = [
        index
        for index, name in enumerate(layer_types)
        if name == "linear_attention"
    ]
    assert len(full_from_types) + len(linear_from_types) == n_decoder_layers, (
        "layer_types contains names other than linear_attention/full_attention"
    )
    full_from_interval = [
        index
        for index in range(n_decoder_layers)
        if index % full_attention_interval == full_attention_interval - 1
    ]
    assert full_from_types == full_from_interval, (
        f"layer_types full indices {full_from_types} != interval {full_from_interval}"
    )
    n_full_layers = len(full_from_types)
    n_linear_layers = len(linear_from_types)

    summary = {
        "authority": AUTHORITY,
        "hidden_size": hidden_size,
        "n_decoder_layers": n_decoder_layers,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "n_mtp_blocks": n_mtp_blocks,
        "full_attention_indices": full_from_types,
        "n_diagrams": 8,
        "n_catalog_nodes": 52,
        "catalog_ids": list(CATALOG_IDS),
        "high_fanout_ids": list(HIGH_FANOUT_IDS),
        "live_across_ids": list(LIVE_ACROSS_IDS),
        "intra_equation_reuse_ids": list(INTRA_EQUATION_REUSE_IDS),
        "shared_weight_ids": list(SHARED_WEIGHT_IDS),
        "state_ids": list(STATE_IDS),
        "regions": list(REGIONS),
        "canonical_sentence": CANONICAL_SENTENCE,
    }
    _assert_identities(summary)
    return summary


def _assert_identities(summary: dict) -> None:
    """Assert dossier identities on a live graph-summary object.

    Args:
        summary: Object produced by :func:`instantiate_graph_summary`.
    """

    assert list(summary) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(summary)} vs {list(SCHEMA_KEYS)}"
    )
    assert summary["authority"] == AUTHORITY
    assert summary["n_catalog_nodes"] == 52
    assert summary["n_diagrams"] == 8
    assert len(summary["catalog_ids"]) == 52
    assert summary["catalog_ids"] == list(CATALOG_IDS)
    assert len(summary["full_attention_indices"]) == 16
    assert summary["full_attention_indices"] == list(LOCKED_FULL_ATTENTION_INDICES)
    assert summary["canonical_sentence"] == CANONICAL_SENTENCE
    assert summary["high_fanout_ids"] == list(HIGH_FANOUT_IDS)
    assert summary["live_across_ids"] == list(LIVE_ACROSS_IDS)
    assert summary["intra_equation_reuse_ids"] == list(INTRA_EQUATION_REUSE_IDS)
    assert summary["shared_weight_ids"] == list(SHARED_WEIGHT_IDS)
    assert summary["state_ids"] == list(STATE_IDS)
    assert summary["regions"] == list(REGIONS)


def load_config(config_path: Path) -> dict:
    """Load Transformers ``config.json`` from ``config_path``.

    Args:
        config_path: Path to the sitting authority config file.

    Returns:
        Parsed JSON object.

    Raises:
        MissingConfig: If the file is absent or unreadable.
        AssertionError: If the file is not a JSON object.
    """

    if not config_path.is_file():
        raise MissingConfig(config_path)
    try:
        text = config_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except OSError as exc:
        raise MissingConfig(config_path) from exc
    assert isinstance(payload, dict), "config.json root is not an object"
    return payload


def _deferred_vision_span(text: str) -> tuple[int, int]:
    """Return the character span of the Deferred vision section.

    Args:
        text: Full markdown document.

    Returns:
        Inclusive-start exclusive-end indices of that section.

    Raises:
        AssertionError: If the heading is missing.
    """

    match = re.search(r"^## Deferred vision\s*$", text, flags=re.MULTILINE)
    assert match is not None, "missing ## Deferred vision heading"
    start = match.start()
    next_heading = HEADING_RE.search(text, match.end())
    end = next_heading.start() if next_heading is not None else len(text)
    return start, end


def diff_summary(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live graph-summary object.

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
    for key in SCHEMA_KEYS:
        if key not in documented or key not in live:
            continue
        if documented[key] != live[key]:
            diffs.append(
                f"{key}: documented {documented[key]!r} != live {live[key]!r}"
            )
    return diffs


def check_dataflow(live: dict, dataflow_path: Path) -> None:
    """Check a dataflow markdown file against a live graph-summary object.

    Args:
        live: Graph-summary object from sitting config.
        dataflow_path: Path to ``dataflow.md``.

    Raises:
        DataflowMismatch: On heading, JSON, diagram, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = dataflow_path.read_text(encoding="utf-8")
    differences: list[str] = []
    if "All MTP-only graph edges and counts are conditional on TASK-02's unverified analysis model." not in text:
        differences.append("missing conditional MTP graph qualification")

    headings = HEADING_RE.findall(text)
    if headings != list(REQUIRED_HEADINGS):
        differences.append(
            "heading mismatch:\n"
            f"  documented: {headings}\n"
            f"  required:   {list(REQUIRED_HEADINGS)}"
        )

    try:
        documented = first_json_fence(text)
    except (AssertionError, json.JSONDecodeError) as exc:
        differences.append(f"json fence: {exc}")
        documented = None

    if documented is not None:
        differences.extend(diff_summary(live, documented))

    mermaid_fences = MERMAID_FENCE_RE.findall(text)
    if len(mermaid_fences) != 8:
        differences.append(
            f"mermaid fence count {len(mermaid_fences)} != 8"
        )
    for index, body in enumerate(mermaid_fences):
        if "flowchart" not in body:
            differences.append(
                f"diagram {index + 1}: mermaid fence does not contain flowchart"
            )
        if index < len(DIAGRAM_REQUIRED_IDS):
            diagram_ids = set(re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\[|\()", body))
            missing_ids = [
                node_id
                for node_id in DIAGRAM_REQUIRED_IDS[index]
                if node_id not in diagram_ids
            ]
            if missing_ids:
                differences.append(
                    f"diagram {index + 1}: missing node ids {missing_ids}"
                )
            if index == 5:
                required_edges = (
                    "K_prior -->|attention read before append| attention",
                    "V_prior -->|attention read before append| attention",
                    "k_rope -->|append current K| K_next",
                    "v_full -->|append current V| V_next",
                    "C_prior -->|three prior taps| fir",
                    "qkv -->|current value| fir",
                    "fir --> C_next",
                    "S_prior --> recurrence",
                    "alpha --> recurrence",
                    "beta --> recurrence",
                    "k_hat --> recurrence",
                    "v_lin --> recurrence",
                    "recurrence --> o",
                    "recurrence --> S_next",
                )
                for edge in required_edges:
                    if edge not in body:
                        differences.append(f"diagram 6: missing exact edge {edge!r}")

    if CANONICAL_SENTENCE not in text:
        differences.append("canonical sentence not present verbatim")

    for node_id in CATALOG_IDS:
        if node_id not in text:
            differences.append(f"catalog id {node_id!r} missing as substring")

    for match in FORBIDDEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        differences.append(f"line {line}: forbidden token {match.group(0)!r}")

    try:
        vis_start, vis_end = _deferred_vision_span(text)
    except AssertionError as exc:
        differences.append(str(exc))
        vis_start, vis_end = 0, 0

    for match in UNKNOWN_RE.finditer(text):
        if vis_start <= match.start() < vis_end:
            continue
        line = text.count("\n", 0, match.start()) + 1
        differences.append(
            f"line {line}: UNKNOWN outside Deferred vision section"
        )

    if differences:
        raise DataflowMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the dataflow checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP logical graph summary from "
            "text_config and check docs/architecture/dataflow.md."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Transformers config.json (text_config authority).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write graph-summary JSON to stdout.",
    )
    parser.add_argument(
        "--dataflow",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the dataflow checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        summary = instantiate_graph_summary(config)
        if args.dataflow is not None:
            if not args.dataflow.is_file():
                print(f"dataflow file not found: {args.dataflow}", file=sys.stderr)
                return 1
            check_dataflow(summary, args.dataflow)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except DataflowMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"assert failed: invalid config JSON: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.config))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.dataflow is None:
        sys.stdout.write(dumps_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
