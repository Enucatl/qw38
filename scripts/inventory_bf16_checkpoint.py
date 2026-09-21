#!/usr/bin/env python3
"""Header-only BF16 checkpoint inventory for the Qwen3.8-27B authority.

Parses safetensor JSON headers (not payloads), maps every tensor onto the
TASK-01 level-2 family taxonomy, and emits machine-checkable totals JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

AUTHORITY = ".cache/authorities/qwen3.8-27b-transformers"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
BF16 = "BF16"
BF16_BYTES = 2
EXPECTED_SHARDS = 18
VISION_BLOCK_TENSORS = 12
VISION_TOTAL_TENSORS = 333

LEVEL1_BY_LEVEL2: dict[str, str] = {
    "embed": "language_embed",
    "final_norm": "language_final_norm",
    "lm_head": "language_lm_head",
    "input_layernorm": "language_layer_norms",
    "post_attention_layernorm": "language_layer_norms",
    "linear_attn.A_log": "language_linear_attn",
    "linear_attn.conv1d": "language_linear_attn",
    "linear_attn.dt_bias": "language_linear_attn",
    "linear_attn.in_proj_a": "language_linear_attn",
    "linear_attn.in_proj_b": "language_linear_attn",
    "linear_attn.in_proj_qkv": "language_linear_attn",
    "linear_attn.in_proj_z": "language_linear_attn",
    "linear_attn.norm": "language_linear_attn",
    "linear_attn.out_proj": "language_linear_attn",
    "self_attn.q_proj": "language_self_attn",
    "self_attn.k_proj": "language_self_attn",
    "self_attn.v_proj": "language_self_attn",
    "self_attn.o_proj": "language_self_attn",
    "self_attn.q_norm": "language_self_attn",
    "self_attn.k_norm": "language_self_attn",
    "mlp.gate_proj": "language_mlp",
    "mlp.up_proj": "language_mlp",
    "mlp.down_proj": "language_mlp",
    "mtp.fc": "mtp",
    "mtp.norm": "mtp",
    "mtp.pre_fc_norm_embedding": "mtp",
    "mtp.pre_fc_norm_hidden": "mtp",
    "mtp.input_layernorm": "mtp",
    "mtp.post_attention_layernorm": "mtp",
    "mtp.self_attn.q_proj": "mtp",
    "mtp.self_attn.k_proj": "mtp",
    "mtp.self_attn.v_proj": "mtp",
    "mtp.self_attn.o_proj": "mtp",
    "mtp.self_attn.q_norm": "mtp",
    "mtp.self_attn.k_norm": "mtp",
    "mtp.mlp.gate_proj": "mtp",
    "mtp.mlp.up_proj": "mtp",
    "mtp.mlp.down_proj": "mtp",
    "vision.patch_embed": "vision",
    "vision.pos_embed": "vision",
    "vision.blocks": "vision",
    "vision.merger": "vision",
}
LEVEL2_IDS: tuple[str, ...] = tuple(LEVEL1_BY_LEVEL2)

EXACT_FAMILIES: dict[str, str] = {
    "model.language_model.embed_tokens.weight": "embed",
    "model.language_model.norm.weight": "final_norm",
    "lm_head.weight": "lm_head",
    "mtp.fc.weight": "mtp.fc",
    "mtp.norm.weight": "mtp.norm",
    "mtp.pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden",
}

LANGUAGE_LAYER_REMAINDERS: dict[str, str] = {
    "input_layernorm.weight": "input_layernorm",
    "post_attention_layernorm.weight": "post_attention_layernorm",
    "linear_attn.A_log": "linear_attn.A_log",
    "linear_attn.conv1d.weight": "linear_attn.conv1d",
    "linear_attn.dt_bias": "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a",
    "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b",
    "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z.weight": "linear_attn.in_proj_z",
    "linear_attn.norm.weight": "linear_attn.norm",
    "linear_attn.out_proj.weight": "linear_attn.out_proj",
    "self_attn.q_proj.weight": "self_attn.q_proj",
    "self_attn.k_proj.weight": "self_attn.k_proj",
    "self_attn.v_proj.weight": "self_attn.v_proj",
    "self_attn.o_proj.weight": "self_attn.o_proj",
    "self_attn.q_norm.weight": "self_attn.q_norm",
    "self_attn.k_norm.weight": "self_attn.k_norm",
    "mlp.gate_proj.weight": "mlp.gate_proj",
    "mlp.up_proj.weight": "mlp.up_proj",
    "mlp.down_proj.weight": "mlp.down_proj",
}

VISION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.visual.patch_embed.", "vision.patch_embed"),
    ("model.visual.pos_embed.", "vision.pos_embed"),
    ("model.visual.blocks.", "vision.blocks"),
    ("model.visual.merger.", "vision.merger"),
)

LANGUAGE_LAYER_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.(.+)$"
)
MTP_LAYER_RE = re.compile(r"^mtp\.layers\.0\.(.+)$")
JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


class MissingCheckpoint(Exception):
    """Raised when the checkpoint directory or a required file is absent."""

    def __init__(self, path: Path) -> None:
        super().__init__(str(path))
        self.path = path


class InventoryMismatch(Exception):
    """Raised when a documented inventory JSON disagrees with a live run."""

    def __init__(self, differences: list[str]) -> None:
        super().__init__("\n".join(differences))
        self.differences = differences


def _numel(shape: list[int]) -> int:
    """Return the product of dimensions in ``shape``."""

    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _read_json_file(path: Path) -> Any:
    """Load UTF-8 JSON from ``path``."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_safetensor_header(path: Path) -> dict[str, Any]:
    """Read one safetensor header JSON object without touching the payload.

    Args:
        path: Shard file whose header should be parsed.

    Returns:
        Decoded header mapping, including ``__metadata__`` when present.

    Raises:
        AssertionError: If the header length or JSON is invalid.
    """

    with path.open("rb") as handle:
        raw_len = handle.read(8)
        assert len(raw_len) == 8, f"{path}: truncated header length"
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = handle.read(header_len)
        assert len(raw_header) == header_len, f"{path}: truncated header JSON"
    header = json.loads(raw_header.decode("utf-8"))
    assert isinstance(header, dict), f"{path}: header JSON is not an object"
    return header


def family_id_for(name: str) -> str:
    """Map a checkpoint tensor name onto exactly one level-2 family id.

    Args:
        name: Full tensor name from a shard header / weight map.

    Returns:
        Level-2 family identifier.

    Raises:
        AssertionError: If ``name`` is unmapped or would match two families.
    """

    matches: list[str] = []
    exact = EXACT_FAMILIES.get(name)
    if exact is not None:
        matches.append(exact)

    layer_match = LANGUAGE_LAYER_RE.fullmatch(name)
    if layer_match is not None:
        remainder = layer_match.group(2)
        family = LANGUAGE_LAYER_REMAINDERS.get(remainder)
        if family is not None:
            matches.append(family)

    mtp_match = MTP_LAYER_RE.fullmatch(name)
    if mtp_match is not None:
        remainder = mtp_match.group(1)
        if remainder.endswith(".weight"):
            remainder = remainder[: -len(".weight")]
        matches.append(f"mtp.{remainder}")

    for prefix, family in VISION_PREFIXES:
        if name.startswith(prefix):
            matches.append(family)

    unique = list(dict.fromkeys(matches))
    assert unique, f"unmapped tensor: {name}"
    assert len(unique) == 1, f"double-mapped tensor {name}: {unique}"
    family_id = unique[0]
    assert family_id in LEVEL1_BY_LEVEL2, f"unknown family id {family_id} for {name}"
    return family_id


def _empty_family(family_id: str) -> dict[str, Any]:
    """Return a zeroed family accumulator for ``family_id``."""

    return {
        "level1": LEVEL1_BY_LEVEL2[family_id],
        "n_tensors": 0,
        "n_parameters": 0,
        "n_bytes": 0,
        "dtype": BF16,
        "shape_set": set(),
    }


def _require_file(path: Path) -> Path:
    """Return ``path`` when it is a file, otherwise raise MissingCheckpoint."""

    if not path.is_file():
        raise MissingCheckpoint(path)
    return path


def _require_dir(path: Path) -> Path:
    """Return ``path`` when it is a directory, otherwise raise MissingCheckpoint."""

    if not path.is_dir():
        raise MissingCheckpoint(path)
    return path


def _sorted_shapes(shape_set: set[tuple[int, ...]]) -> list[list[int]]:
    """Return unique shapes as sorted arrays of ints."""

    return [list(shape) for shape in sorted(shape_set)]


def _is_json_int(value: Any) -> bool:
    """Return True when ``value`` is a JSON integer (not bool or float)."""

    return isinstance(value, int) and not isinstance(value, bool)


def inventory_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Build totals JSON for ``checkpoint`` and run fail-closed asserts.

    Args:
        checkpoint: Directory containing config, index, and shard files.

    Returns:
        Totals object matching the TASK-01 JSON schema.

    Raises:
        MissingCheckpoint: If the directory, index, config, or a shard is absent.
        AssertionError: If mapping, dtype, totals, shapes, or occupancy fail.
    """

    _require_dir(checkpoint)
    config_path = _require_file(checkpoint / CONFIG_NAME)
    index_path = _require_file(checkpoint / INDEX_NAME)
    config = _read_json_file(config_path)
    index = _read_json_file(index_path)
    assert isinstance(config, dict), "config.json is not an object"
    assert isinstance(index, dict), "model.safetensors.index.json is not an object"
    text_config = config["text_config"]
    vision_config = config["vision_config"]
    assert isinstance(text_config, dict), "text_config is not an object"
    assert isinstance(vision_config, dict), "vision_config is not an object"

    weight_map = index["weight_map"]
    metadata = index["metadata"]
    assert isinstance(weight_map, dict), "weight_map is not an object"
    assert isinstance(metadata, dict), "index metadata is not an object"
    raw_total_size = metadata["total_size"]
    index_metadata_total_size = int(raw_total_size)
    assert float(raw_total_size) == float(index_metadata_total_size), (
        f"index metadata.total_size is not integral: {raw_total_size!r}"
    )

    shard_names = sorted(set(weight_map.values()))
    shard_tensors: dict[str, dict[str, Any]] = {}
    for shard_name in shard_names:
        shard_path = _require_file(checkpoint / shard_name)
        header = _parse_safetensor_header(shard_path)
        header.pop("__metadata__", None)
        shard_tensors[shard_name] = header

    header_names: set[str] = set()
    tensors: dict[str, dict[str, Any]] = {}
    for shard_name, header in shard_tensors.items():
        for name, spec in header.items():
            assert name not in header_names, f"duplicate header tensor {name}"
            header_names.add(name)
            assert isinstance(spec, dict), f"{name}: header entry is not an object"
            dtype = spec["dtype"]
            shape = [int(dim) for dim in spec["shape"]]
            offsets = spec["data_offsets"]
            assert len(offsets) == 2, f"{name}: data_offsets is not a pair"
            start, end = int(offsets[0]), int(offsets[1])
            n_parameters = _numel(shape)
            n_bytes = end - start
            assert dtype == BF16, f"{name}: dtype {dtype!r} is not {BF16}"
            assert n_bytes == n_parameters * BF16_BYTES, (
                f"{name}: byte size {n_bytes} != numel {n_parameters} * {BF16_BYTES}"
            )
            tensors[name] = {
                "dtype": dtype,
                "shape": shape,
                "n_parameters": n_parameters,
                "n_bytes": n_bytes,
                "shard": shard_name,
            }

    assert header_names == set(weight_map.keys()), (
        "header names disagree with weight_map keys: "
        f"only_headers={sorted(header_names - set(weight_map))} "
        f"only_index={sorted(set(weight_map) - header_names)}"
    )
    assert len(shard_names) == EXPECTED_SHARDS, (
        f"expected {EXPECTED_SHARDS} shards, found {len(shard_names)}"
    )

    families = {family_id: _empty_family(family_id) for family_id in LEVEL2_IDS}
    mapped: dict[str, str] = {}
    for name, spec in tensors.items():
        family_id = family_id_for(name)
        mapped[name] = family_id
        family = families[family_id]
        family["n_tensors"] += 1
        family["n_parameters"] += spec["n_parameters"]
        family["n_bytes"] += spec["n_bytes"]
        family["shape_set"].add(tuple(spec["shape"]))
        assert family["dtype"] == BF16
        assert spec["dtype"] == BF16

    n_parameters = sum(spec["n_parameters"] for spec in tensors.values())
    n_bytes = sum(spec["n_bytes"] for spec in tensors.values())
    assert n_bytes == n_parameters * BF16_BYTES == index_metadata_total_size, (
        "byte/parameter/index totals disagree: "
        f"n_bytes={n_bytes} n_parameters={n_parameters} "
        f"index_metadata_total_size={index_metadata_total_size}"
    )

    _assert_layer_types(text_config, mapped)
    _assert_shapes(text_config, families)
    _assert_occupancy(text_config, vision_config, families)

    family_json: dict[str, Any] = {}
    for family_id in LEVEL2_IDS:
        family = families[family_id]
        family_json[family_id] = {
            "level1": family["level1"],
            "n_tensors": int(family["n_tensors"]),
            "n_parameters": int(family["n_parameters"]),
            "n_bytes": int(family["n_bytes"]),
            "dtype": BF16,
            "shapes": _sorted_shapes(family["shape_set"]),
        }

    return {
        "authority": AUTHORITY,
        "n_tensors": int(len(tensors)),
        "n_shards": int(len(shard_names)),
        "dtype": BF16,
        "n_parameters": int(n_parameters),
        "n_bytes": int(n_bytes),
        "index_metadata_total_size": int(index_metadata_total_size),
        "families": family_json,
    }


def _assert_layer_types(text_config: dict[str, Any], mapped: dict[str, str]) -> None:
    """Assert ``layer_types`` length, pattern, and per-layer module exclusivity."""

    layer_types = text_config["layer_types"]
    n_layers = int(text_config["num_hidden_layers"])
    assert isinstance(layer_types, list)
    assert len(layer_types) == n_layers == 64, (
        f"layer_types length {len(layer_types)} vs num_hidden_layers {n_layers}"
    )
    n_linear = sum(1 for kind in layer_types if kind == "linear_attention")
    n_full = sum(1 for kind in layer_types if kind == "full_attention")
    assert n_linear == 48, f"linear_attention count {n_linear} != 48"
    assert n_full == 16, f"full_attention count {n_full} != 16"
    for index, kind in enumerate(layer_types):
        expected = "full_attention" if index % 4 == 3 else "linear_attention"
        assert kind == expected, (
            f"layer_types[{index}]={kind!r} expected {expected!r}"
        )

    layer_has_linear: dict[int, bool] = {i: False for i in range(n_layers)}
    layer_has_self: dict[int, bool] = {i: False for i in range(n_layers)}
    for name in mapped:
        match = LANGUAGE_LAYER_RE.fullmatch(name)
        if match is None:
            continue
        index = int(match.group(1))
        remainder = match.group(2)
        if remainder.startswith("linear_attn."):
            layer_has_linear[index] = True
        elif remainder.startswith("self_attn."):
            layer_has_self[index] = True

    for index, kind in enumerate(layer_types):
        has_linear = layer_has_linear[index]
        has_self = layer_has_self[index]
        assert has_linear ^ has_self, (
            f"layer {index} must have linear_attn xor self_attn "
            f"(linear={has_linear} self={has_self})"
        )
        if kind == "linear_attention":
            assert has_linear and not has_self, (
                f"layer {index} is linear_attention but modules "
                f"linear={has_linear} self={has_self}"
            )
        else:
            assert kind == "full_attention"
            assert has_self and not has_linear, (
                f"layer {index} is full_attention but modules "
                f"linear={has_linear} self={has_self}"
            )


def _unique_shape(family: dict[str, Any], family_id: str) -> list[int]:
    """Return the single unique shape recorded for ``family_id``."""

    shapes = _sorted_shapes(family["shape_set"])
    assert len(shapes) == 1, f"{family_id} has shapes {shapes}"
    return shapes[0]


def _assert_shape(family: dict[str, Any], family_id: str, expected: list[int]) -> None:
    """Assert ``family_id`` has exactly ``expected`` as its unique shape."""

    actual = _unique_shape(family, family_id)
    assert actual == expected, f"{family_id} shape {actual} != {expected}"


def _assert_shapes(text_config: dict[str, Any], families: dict[str, Any]) -> None:
    """Assert family shapes against ``text_config`` dimensions."""

    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])
    vocab = int(text_config["vocab_size"])
    n_heads = int(text_config["num_attention_heads"])
    n_kv = int(text_config["num_key_value_heads"])
    head_dim = int(text_config["head_dim"])
    linear_key_heads = int(text_config["linear_num_key_heads"])
    linear_value_heads = int(text_config["linear_num_value_heads"])
    linear_key_dim = int(text_config["linear_key_head_dim"])
    linear_value_dim = int(text_config["linear_value_head_dim"])
    conv_kernel = int(text_config["linear_conv_kernel_dim"])
    qkv_width = (linear_key_heads + linear_key_heads + linear_value_heads) * linear_key_dim
    z_width = linear_value_heads * linear_value_dim

    _assert_shape(families["embed"], "embed", [vocab, hidden])
    _assert_shape(families["lm_head"], "lm_head", [vocab, hidden])
    _assert_shape(families["final_norm"], "final_norm", [hidden])
    _assert_shape(families["input_layernorm"], "input_layernorm", [hidden])
    _assert_shape(
        families["post_attention_layernorm"],
        "post_attention_layernorm",
        [hidden],
    )
    _assert_shape(
        families["mlp.gate_proj"], "mlp.gate_proj", [intermediate, hidden]
    )
    _assert_shape(families["mlp.up_proj"], "mlp.up_proj", [intermediate, hidden])
    _assert_shape(
        families["mlp.down_proj"], "mlp.down_proj", [hidden, intermediate]
    )

    q_out = 2 * n_heads * head_dim
    kv_out = n_kv * head_dim
    o_in = n_heads * head_dim
    for prefix in ("self_attn", "mtp.self_attn"):
        _assert_shape(families[f"{prefix}.q_proj"], f"{prefix}.q_proj", [q_out, hidden])
        _assert_shape(families[f"{prefix}.k_proj"], f"{prefix}.k_proj", [kv_out, hidden])
        _assert_shape(families[f"{prefix}.v_proj"], f"{prefix}.v_proj", [kv_out, hidden])
        _assert_shape(families[f"{prefix}.o_proj"], f"{prefix}.o_proj", [hidden, o_in])
        _assert_shape(families[f"{prefix}.q_norm"], f"{prefix}.q_norm", [head_dim])
        _assert_shape(families[f"{prefix}.k_norm"], f"{prefix}.k_norm", [head_dim])

    _assert_shape(
        families["linear_attn.in_proj_qkv"],
        "linear_attn.in_proj_qkv",
        [qkv_width, hidden],
    )
    _assert_shape(
        families["linear_attn.in_proj_z"],
        "linear_attn.in_proj_z",
        [z_width, hidden],
    )
    _assert_shape(
        families["linear_attn.out_proj"],
        "linear_attn.out_proj",
        [hidden, z_width],
    )
    _assert_shape(
        families["linear_attn.conv1d"],
        "linear_attn.conv1d",
        [qkv_width, 1, conv_kernel],
    )
    _assert_shape(
        families["linear_attn.A_log"], "linear_attn.A_log", [linear_value_heads]
    )
    _assert_shape(
        families["linear_attn.dt_bias"],
        "linear_attn.dt_bias",
        [linear_value_heads],
    )
    _assert_shape(
        families["linear_attn.in_proj_a"],
        "linear_attn.in_proj_a",
        [linear_value_heads, hidden],
    )
    _assert_shape(
        families["linear_attn.in_proj_b"],
        "linear_attn.in_proj_b",
        [linear_value_heads, hidden],
    )
    _assert_shape(
        families["linear_attn.norm"], "linear_attn.norm", [linear_value_dim]
    )
    _assert_shape(families["mtp.fc"], "mtp.fc", [hidden, 2 * hidden])
    _assert_shape(families["mtp.norm"], "mtp.norm", [hidden])
    _assert_shape(
        families["mtp.pre_fc_norm_embedding"],
        "mtp.pre_fc_norm_embedding",
        [hidden],
    )
    _assert_shape(
        families["mtp.pre_fc_norm_hidden"],
        "mtp.pre_fc_norm_hidden",
        [hidden],
    )
    _assert_shape(
        families["mtp.input_layernorm"], "mtp.input_layernorm", [hidden]
    )
    _assert_shape(
        families["mtp.post_attention_layernorm"],
        "mtp.post_attention_layernorm",
        [hidden],
    )
    _assert_shape(
        families["mtp.mlp.gate_proj"],
        "mtp.mlp.gate_proj",
        [intermediate, hidden],
    )
    _assert_shape(
        families["mtp.mlp.up_proj"], "mtp.mlp.up_proj", [intermediate, hidden]
    )
    _assert_shape(
        families["mtp.mlp.down_proj"],
        "mtp.mlp.down_proj",
        [hidden, intermediate],
    )

    language_mlp_norms = (
        ("mlp.gate_proj", "mtp.mlp.gate_proj"),
        ("mlp.up_proj", "mtp.mlp.up_proj"),
        ("mlp.down_proj", "mtp.mlp.down_proj"),
        ("input_layernorm", "mtp.input_layernorm"),
        ("post_attention_layernorm", "mtp.post_attention_layernorm"),
        ("self_attn.q_proj", "mtp.self_attn.q_proj"),
        ("self_attn.k_proj", "mtp.self_attn.k_proj"),
        ("self_attn.v_proj", "mtp.self_attn.v_proj"),
        ("self_attn.o_proj", "mtp.self_attn.o_proj"),
        ("self_attn.q_norm", "mtp.self_attn.q_norm"),
        ("self_attn.k_norm", "mtp.self_attn.k_norm"),
    )
    for language_id, mtp_id in language_mlp_norms:
        language_shape = _unique_shape(families[language_id], language_id)
        mtp_shape = _unique_shape(families[mtp_id], mtp_id)
        assert language_shape == mtp_shape, (
            f"{mtp_id} shape {mtp_shape} != language {language_id} {language_shape}"
        )


def _assert_occupancy(
    text_config: dict[str, Any],
    vision_config: dict[str, Any],
    families: dict[str, Any],
) -> None:
    """Assert per-family tensor counts from config-derived occupancy."""

    n_linear = 48
    n_full = 16
    n_layers = int(text_config["num_hidden_layers"])
    vision_depth = int(vision_config["depth"])
    for family_id, family in families.items():
        if family_id.startswith("linear_attn."):
            assert family["n_tensors"] == n_linear, (
                f"{family_id} n_tensors={family['n_tensors']} != {n_linear}"
            )
        elif family_id.startswith("self_attn."):
            assert family["n_tensors"] == n_full, (
                f"{family_id} n_tensors={family['n_tensors']} != {n_full}"
            )
        elif family_id.startswith("mlp.") or family_id in {
            "input_layernorm",
            "post_attention_layernorm",
        }:
            assert family["n_tensors"] == n_layers, (
                f"{family_id} n_tensors={family['n_tensors']} != {n_layers}"
            )
        elif family_id.startswith("mtp."):
            assert family["n_tensors"] == 1, (
                f"{family_id} n_tensors={family['n_tensors']} != 1"
            )
    assert families["vision.blocks"]["n_tensors"] == vision_depth * VISION_BLOCK_TENSORS, (
        "vision.blocks occupancy "
        f"{families['vision.blocks']['n_tensors']} != "
        f"{vision_depth}*{VISION_BLOCK_TENSORS}"
    )
    vision_total = sum(
        families[family_id]["n_tensors"]
        for family_id in LEVEL2_IDS
        if LEVEL1_BY_LEVEL2[family_id] == "vision"
    )
    assert vision_total == VISION_TOTAL_TENSORS, (
        f"vision total {vision_total} != {VISION_TOTAL_TENSORS}"
    )
    assert families["embed"]["n_tensors"] == 1
    assert families["final_norm"]["n_tensors"] == 1
    assert families["lm_head"]["n_tensors"] == 1


def dumps_totals(totals: dict[str, Any]) -> str:
    """Pretty-print totals JSON with script emission key order.

    Args:
        totals: Object produced by :func:`inventory_checkpoint`.

    Returns:
        Pretty-printed JSON including a trailing newline.
    """

    return json.dumps(totals, indent=2) + "\n"


def first_json_fence(text: str) -> dict[str, Any]:
    """Parse the first fenced ``json`` code block in ``text``.

    Args:
        text: Markdown (or other) document containing a `` ```json `` fence.

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


def _json_int_fields(obj: dict[str, Any], path: str, fields: tuple[str, ...]) -> list[str]:
    """Return diffs for required integer fields on ``obj``."""

    diffs: list[str] = []
    for field in fields:
        value = obj.get(field)
        if not _is_json_int(value):
            diffs.append(f"{path}.{field}: expected int, got {value!r}")
    return diffs


def diff_inventory(live: dict[str, Any], documented: dict[str, Any]) -> list[str]:
    """Compare documented inventory JSON against a live totals object.

    Args:
        live: Fresh script inventory.
        documented: Object parsed from a markdown JSON fence.

    Returns:
        Human-readable difference strings; empty when the documents match.
    """

    diffs: list[str] = []
    top_keys = (
        "authority",
        "n_tensors",
        "n_shards",
        "dtype",
        "n_parameters",
        "n_bytes",
        "index_metadata_total_size",
        "families",
    )
    live_keys = set(live)
    doc_keys = set(documented)
    for extra in sorted(doc_keys - set(top_keys)):
        diffs.append(f"documented extra top-level key: {extra}")
    for missing in sorted(set(top_keys) - doc_keys):
        diffs.append(f"documented missing top-level key: {missing}")
    for extra in sorted(live_keys - doc_keys):
        diffs.append(f"live extra top-level key: {extra}")
    for missing in sorted(doc_keys - live_keys):
        diffs.append(f"live missing documented top-level key: {missing}")

    for key in ("authority", "dtype"):
        if documented.get(key) != live.get(key):
            diffs.append(
                f"{key}: documented={documented.get(key)!r} live={live.get(key)!r}"
            )
    diffs.extend(
        _json_int_fields(
            documented,
            "$",
            (
                "n_tensors",
                "n_shards",
                "n_parameters",
                "n_bytes",
                "index_metadata_total_size",
            ),
        )
    )
    for key in (
        "n_tensors",
        "n_shards",
        "n_parameters",
        "n_bytes",
        "index_metadata_total_size",
    ):
        if documented.get(key) != live.get(key):
            diffs.append(
                f"{key}: documented={documented.get(key)!r} live={live.get(key)!r}"
            )

    doc_families = documented.get("families")
    live_families = live["families"]
    if not isinstance(doc_families, dict):
        diffs.append("families: documented value is not an object")
        return diffs

    doc_family_ids = set(doc_families)
    live_family_ids = set(live_families)
    for extra in sorted(doc_family_ids - live_family_ids):
        diffs.append(f"families extra documented id: {extra}")
    for missing in sorted(live_family_ids - doc_family_ids):
        diffs.append(f"families missing documented id: {missing}")

    family_int_fields = ("n_tensors", "n_parameters", "n_bytes")
    for family_id in LEVEL2_IDS:
        if family_id not in doc_families or family_id not in live_families:
            continue
        doc_family = doc_families[family_id]
        live_family = live_families[family_id]
        path = f"families.{family_id}"
        if not isinstance(doc_family, dict):
            diffs.append(f"{path}: documented value is not an object")
            continue
        expected_keys = {"level1", "n_tensors", "n_parameters", "n_bytes", "dtype", "shapes"}
        extras = sorted(set(doc_family) - expected_keys)
        missing = sorted(expected_keys - set(doc_family))
        for extra in extras:
            diffs.append(f"{path} extra key: {extra}")
        for key in missing:
            diffs.append(f"{path} missing key: {key}")
        if doc_family.get("level1") != live_family["level1"]:
            diffs.append(
                f"{path}.level1: documented={doc_family.get('level1')!r} "
                f"live={live_family['level1']!r}"
            )
        if doc_family.get("dtype") != live_family["dtype"]:
            diffs.append(
                f"{path}.dtype: documented={doc_family.get('dtype')!r} "
                f"live={live_family['dtype']!r}"
            )
        diffs.extend(_json_int_fields(doc_family, path, family_int_fields))
        for field in family_int_fields:
            if doc_family.get(field) != live_family[field]:
                diffs.append(
                    f"{path}.{field}: documented={doc_family.get(field)!r} "
                    f"live={live_family[field]!r}"
                )
        doc_shapes = doc_family.get("shapes")
        live_shapes = live_family["shapes"]
        if doc_shapes != live_shapes:
            diffs.append(
                f"{path}.shapes: documented={doc_shapes!r} live={live_shapes!r}"
            )
    return diffs


def check_inventory(live: dict[str, Any], inventory_path: Path) -> None:
    """Compare the first JSON fence in ``inventory_path`` to ``live``.

    Args:
        live: Fresh script inventory.
        inventory_path: Markdown file containing the totals fence.

    Raises:
        InventoryMismatch: When documented JSON disagrees with ``live``.
        AssertionError: When the file has no usable JSON fence.
        OSError: When the file cannot be read.
    """

    text = inventory_path.read_text(encoding="utf-8")
    documented = first_json_fence(text)
    differences = diff_inventory(live, documented)
    if differences:
        raise InventoryMismatch(differences)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the inventory tool."""

    parser = argparse.ArgumentParser(
        description=(
            "Inventory the Qwen3.8-27B BF16 Transformers checkpoint from "
            "safetensor headers (payloads unread)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory (config.json, index, and 18 shards).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write totals JSON to stdout (default when not checking inventory).",
    )
    parser.add_argument(
        "--check-inventory",
        type=Path,
        default=None,
        dest="check_inventory",
        help="Markdown path whose first json fence must match a live inventory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the inventory CLI.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 success, 1 content/assert/mismatch, 2 missing path.
    """

    args = _parse_args(argv)
    try:
        totals = inventory_checkpoint(args.checkpoint)
        if args.check_inventory is not None:
            if not args.check_inventory.is_file():
                print(
                    f"inventory file not found: {args.check_inventory}",
                    file=sys.stderr,
                )
                return 1
            check_inventory(totals, args.check_inventory)
    except MissingCheckpoint as exc:
        print(str(exc.path), file=sys.stderr)
        return 2
    except InventoryMismatch as exc:
        print("\n".join(exc.differences), file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"assert failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        missing = Path(getattr(exc, "filename", args.checkpoint))
        print(str(missing), file=sys.stderr)
        return 2

    if args.json or args.check_inventory is None:
        sys.stdout.write(dumps_totals(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
