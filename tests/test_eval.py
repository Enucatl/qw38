from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tools.qw38_eval import (
    EvalError,
    LogitsRequest,
    TraceRequest,
    parse_tokens,
    read_logits_result,
)


def test_parse_tokens_is_strict_and_vocab_bounded() -> None:
    assert parse_tokens("0,42,248319") == (0, 42, 248319)
    for value in ("", "1,,2", "1, 2", "-1", "248320"):
        with pytest.raises(EvalError):
            parse_tokens(value)


def test_request_requires_explicit_source_identity(tmp_path: Path) -> None:
    with pytest.raises(EvalError):
        LogitsRequest(tmp_path / "model", (1,), tmp_path / "out", "", "clean")
    with pytest.raises(EvalError):
        LogitsRequest(tmp_path / "model", (1,), tmp_path / "out", "rev", "unknown")


def test_trace_request_requires_distinct_pinned_filters(tmp_path: Path) -> None:
    common = (tmp_path / "model", (1,), tmp_path / "out", "rev", "clean")
    with pytest.raises(EvalError):
        TraceRequest(*common, ())
    with pytest.raises(EvalError):
        TraceRequest(*common, ("0:layer_residual", "0:layer_residual"))
    with pytest.raises(EvalError):
        TraceRequest(*common, ("1:layer_residual",))


def test_logits_reader_rejects_wrong_blob_digest(tmp_path: Path) -> None:
    blob = struct.pack("<248320f", *([0.0] * 248320))
    (tmp_path / "logits.f32le.bin").write_bytes(blob)
    (tmp_path / "result.json").write_text(
        '{"schema":"qw38.eval-result","version":1,"mode":"logits",'
        '"status":"ok","tokens":[1],"positions":[0],"frontier":1,'
        '"logits":{"dtype":"f32-le","shape":[248320],"byte_count":993280,'
        '"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}'
    )
    with pytest.raises(EvalError):
        read_logits_result(tmp_path)
