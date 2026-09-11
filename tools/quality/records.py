"""Inspectable per-case JSON/TSV evidence records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.quality.compare import records_to_tsv


def case_evidence(
    *,
    case_id: str,
    suite_class: str,
    quartz: Mapping[str, Any] | None,
    llama: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": case_id,
        "suite_class": suite_class,
        "quartz": dict(quartz) if quartz is not None else None,
        "llama": dict(llama) if llama is not None else None,
        "inspectable": True,
        "aggregate_only": False,
    }
    if quartz is not None and llama is not None:
        record["delta_nll"] = float(quartz["nll"]) - float(llama["nll"])
        record["delta_avg_nll"] = float(quartz["avg_nll"]) - float(llama["avg_nll"])
    if extra:
        record.update(dict(extra))
    return record


def write_case_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def write_case_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(records_to_tsv(rows), encoding="utf-8")
