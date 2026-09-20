#!/usr/bin/env python3
"""Check Qwen3.8-27B model-semantics dimensions against sitting text_config.

Reads ``text_config`` from a Transformers ``config.json`` (no safetensor
payloads) and either prints the instantiated-dimensions object or checks that
``docs/architecture/model-semantics.md`` matches it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"

SCHEMA_KEYS: tuple[str, ...] = (
    "authority",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "gqa_group_size",
    "rotary_dim",
    "n_rope_freq",
    "mrope_section",
    "mrope_section_sum",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_kv_repeat",
    "linear_qkv_width",
    "linear_z_width",
    "linear_conv_kernel_dim",
    "linear_conv_delay",
    "linear_state_heads",
    "linear_state_dk",
    "linear_state_dv",
    "full_attention_interval",
    "n_linear_layers",
    "n_full_layers",
    "full_attention_indices",
    "mtp_num_hidden_layers",
)

REQUIRED_HEADINGS: tuple[str, ...] = (
    "Authority",
    "Notation",
    "Embedding",
    "Normalization",
    "Residual decoder layer",
    "Full attention (Gated Attention)",
    "Rotary embeddings (partial mRoPE)",
    "Linear attention (Gated DeltaNet)",
    "MLP",
    "Persistent state",
    "Primary logits",
    "MTP",
    "Algebraic equivalents",
    "Deferred vision",
    "Instantiated dimensions",
)

JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)
FORBIDDEN_RE = re.compile(r"TBD|TODO|\?\?\?")
UNKNOWN_RE = re.compile(r"UNKNOWN")


class MissingConfig(Exception):
    """Raised when the config path is absent or unreadable."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class SemanticsMismatch(Exception):
    """Raised when the semantics markdown fails a content check."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def dumps_totals(totals: dict) -> str:
    """Pretty-print the instantiated-dimensions object.

    Args:
        totals: Object produced by :func:`instantiate_dimensions`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(totals, indent=2) + "\n"


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


def _require_divides(numerator: int, denominator: int, name: str) -> int:
    """Return ``numerator // denominator`` after asserting exact division.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        name: Identity name used in the assertion message.

    Returns:
        Exact integer quotient.
    """

    assert denominator != 0, f"{name}: divisor is 0"
    assert numerator % denominator == 0, (
        f"{name}: {numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def instantiate_dimensions(config: dict) -> dict:
    """Compute the TASK-02 instantiated-dimensions object from ``config``.

    Args:
        config: Parsed Transformers ``config.json`` object.

    Returns:
        Ordered totals dict matching the dossier JSON schema.

    Raises:
        AssertionError: If required fields are missing or identities fail.
    """

    assert isinstance(config, dict), "config.json root is not an object"
    assert "text_config" in config, "config.json has no text_config"
    text = config["text_config"]
    assert isinstance(text, dict), "text_config is not an object"

    hidden_size = _require_int(text, "hidden_size")
    intermediate_size = _require_int(text, "intermediate_size")
    vocab_size = _require_int(text, "vocab_size")
    num_hidden_layers = _require_int(text, "num_hidden_layers")
    num_attention_heads = _require_int(text, "num_attention_heads")
    num_key_value_heads = _require_int(text, "num_key_value_heads")
    head_dim = _require_int(text, "head_dim")
    linear_num_key_heads = _require_int(text, "linear_num_key_heads")
    linear_num_value_heads = _require_int(text, "linear_num_value_heads")
    linear_key_head_dim = _require_int(text, "linear_key_head_dim")
    linear_value_head_dim = _require_int(text, "linear_value_head_dim")
    linear_conv_kernel_dim = _require_int(text, "linear_conv_kernel_dim")
    full_attention_interval = _require_int(text, "full_attention_interval")
    mtp_num_hidden_layers = _require_int(text, "mtp_num_hidden_layers")

    assert "partial_rotary_factor" in text, "missing partial_rotary_factor"
    partial_rotary_factor = text["partial_rotary_factor"]
    rotary_dim_f = head_dim * partial_rotary_factor
    assert float(rotary_dim_f).is_integer(), (
        f"rotary_dim is not integral: {rotary_dim_f}"
    )
    rotary_dim = int(rotary_dim_f)
    n_rope_freq = _require_divides(rotary_dim, 2, "n_rope_freq")

    assert "rope_parameters" in text, "missing rope_parameters"
    rope = text["rope_parameters"]
    assert isinstance(rope, dict), "rope_parameters is not an object"
    assert "mrope_section" in rope, "missing mrope_section"
    mrope_section = rope["mrope_section"]
    assert isinstance(mrope_section, list), "mrope_section is not a list"
    assert all(isinstance(item, int) and not isinstance(item, bool) for item in mrope_section), (
        f"mrope_section is not a list of ints: {mrope_section!r}"
    )
    mrope_section = [int(item) for item in mrope_section]
    mrope_section_sum = sum(mrope_section)

    gqa_group_size = _require_divides(
        num_attention_heads, num_key_value_heads, "gqa_group_size"
    )
    linear_kv_repeat = _require_divides(
        linear_num_value_heads, linear_num_key_heads, "linear_kv_repeat"
    )
    linear_qkv_width = (
        linear_num_key_heads * linear_key_head_dim
        + linear_num_key_heads * linear_key_head_dim
        + linear_num_value_heads * linear_value_head_dim
    )
    linear_z_width = linear_num_value_heads * linear_value_head_dim
    linear_conv_delay = linear_conv_kernel_dim - 1
    linear_state_heads = linear_num_value_heads
    linear_state_dk = linear_key_head_dim
    linear_state_dv = linear_value_head_dim

    assert "layer_types" in text, "missing layer_types"
    layer_types = text["layer_types"]
    assert isinstance(layer_types, list), "layer_types is not a list"
    assert len(layer_types) == num_hidden_layers, (
        f"layer_types length {len(layer_types)} != num_hidden_layers {num_hidden_layers}"
    )
    full_from_types = [
        index for index, name in enumerate(layer_types) if name == "full_attention"
    ]
    linear_from_types = [
        index for index, name in enumerate(layer_types) if name == "linear_attention"
    ]
    assert len(full_from_types) + len(linear_from_types) == num_hidden_layers, (
        "layer_types contains names other than linear_attention/full_attention"
    )
    full_from_interval = [
        index
        for index in range(num_hidden_layers)
        if index % full_attention_interval == full_attention_interval - 1
    ]
    assert full_from_types == full_from_interval, (
        f"layer_types full indices {full_from_types} != interval {full_from_interval}"
    )
    n_full_layers = len(full_from_types)
    n_linear_layers = len(linear_from_types)

    totals = {
        "authority": AUTHORITY,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "gqa_group_size": gqa_group_size,
        "rotary_dim": rotary_dim,
        "n_rope_freq": n_rope_freq,
        "mrope_section": mrope_section,
        "mrope_section_sum": mrope_section_sum,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_kv_repeat": linear_kv_repeat,
        "linear_qkv_width": linear_qkv_width,
        "linear_z_width": linear_z_width,
        "linear_conv_kernel_dim": linear_conv_kernel_dim,
        "linear_conv_delay": linear_conv_delay,
        "linear_state_heads": linear_state_heads,
        "linear_state_dk": linear_state_dk,
        "linear_state_dv": linear_state_dv,
        "full_attention_interval": full_attention_interval,
        "n_linear_layers": n_linear_layers,
        "n_full_layers": n_full_layers,
        "full_attention_indices": full_from_types,
        "mtp_num_hidden_layers": mtp_num_hidden_layers,
    }
    _assert_identities(totals, text)
    return totals


def _assert_identities(totals: dict, text: dict) -> None:
    """Assert dossier identities on a live instantiated object.

    Args:
        totals: Object produced by :func:`instantiate_dimensions`.
        text: The ``text_config`` mapping used to build ``totals``.
    """

    assert list(totals) == list(SCHEMA_KEYS), (
        f"schema key order mismatch: {list(totals)} vs {list(SCHEMA_KEYS)}"
    )
    assert totals["authority"] == AUTHORITY
    assert totals["rotary_dim"] == 64
    assert totals["n_rope_freq"] == 32
    assert totals["mrope_section"] == [11, 11, 10]
    assert totals["mrope_section_sum"] == 32
    assert totals["mrope_section_sum"] == totals["n_rope_freq"]
    assert totals["linear_qkv_width"] == 10240
    assert totals["linear_z_width"] == 6144
    assert totals["linear_kv_repeat"] == 3
    assert totals["gqa_group_size"] == 6
    assert totals["full_attention_indices"] == list(range(3, 64, 4))
    assert totals["n_full_layers"] == 16
    assert totals["n_linear_layers"] == 48
    assert totals["linear_conv_delay"] == 3
    scale = totals["head_dim"] ** -0.5
    assert scale == 1 / 16, f"attention scale {scale} != 1/16"
    assert text.get("attn_output_gate") is True
    assert text.get("output_gate_type") == "swish"
    assert text.get("mamba_ssm_dtype") == "float32"
    assert text.get("mtp_use_dedicated_embeddings") is False
    assert text.get("tie_word_embeddings") is False
    assert text.get("attention_bias") is False
    assert text.get("use_cache") is True


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


def check_semantics(live: dict, semantics_path: Path) -> None:
    """Check a semantics markdown file against a live dimensions object.

    Args:
        live: Instantiated-dimensions object from sitting config.
        semantics_path: Path to ``model-semantics.md``.

    Raises:
        SemanticsMismatch: On heading, JSON, or token mismatches.
        OSError: If the file cannot be read.
    """

    text = semantics_path.read_text(encoding="utf-8")
    differences: list[str] = []

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
        differences.extend(diff_totals(live, documented))

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
        raise SemanticsMismatch(differences)


def diff_totals(live: dict, documented: dict) -> list[str]:
    """Compare documented JSON against a live instantiated object.

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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the semantics checker."""

    parser = argparse.ArgumentParser(
        description=(
            "Instantiate Qwen3.8-27B language+MTP dimensions from text_config "
            "and check docs/architecture/model-semantics.md."
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
        help="Write instantiated-dimensions JSON to stdout.",
    )
    parser.add_argument(
        "--semantics",
        type=Path,
        default=None,
        help="Markdown path whose first json fence must match live dimensions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the model-semantics checker CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing config.
    """

    args = _parse_args(argv)
    try:
        config = load_config(args.config)
        totals = instantiate_dimensions(config)
        if args.semantics is not None:
            if not args.semantics.is_file():
                print(f"semantics file not found: {args.semantics}", file=sys.stderr)
                return 1
            check_semantics(totals, args.semantics)
    except MissingConfig as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except SemanticsMismatch as exc:
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

    if args.json or args.semantics is None:
        sys.stdout.write(dumps_totals(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
