"""Mechanical checks for the code-linked documentation contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "implementation_ledger.md"
AUDIT = ROOT / "docs/65-documentation-audit.md"
HANDBOOK = ROOT / "docs/README.md"

TASK_ID = re.compile(r"\b[A-Z]{3}-\d{3}\b")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
CHAPTER_FILE = re.compile(r"(?:^|[/(])((?:1[3-9]|[2-5]\d|6[0-5])-[^/)]+\.md)")
CLAIM_LABELS = ("Measured", "External", "Estimated", "Proposed")


def _table_rows(text: str) -> list[list[str]]:
    """Return cells from Markdown pipe-table rows."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < 2:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _ledger_tasks() -> dict[str, tuple[str, str]]:
    """Read task descriptions and statuses from the ledger's task table."""
    tasks: dict[str, tuple[str, str]] = {}
    for row in _table_rows(LEDGER.read_text(encoding="utf-8")):
        if len(row) < 5:
            continue
        match = TASK_ID.fullmatch(row[0])
        if match:
            tasks[row[0]] = (row[1], row[3].lower())
    return tasks


def _audit_rows() -> dict[str, str]:
    """Collect audit-table rows keyed by each task ID they mention."""
    rows: dict[str, str] = {}
    for row in _table_rows(AUDIT.read_text(encoding="utf-8")):
        ids = TASK_ID.findall(" ".join(row))
        for task_id in ids:
            if task_id in rows:
                raise AssertionError(f"task {task_id} occurs more than once in audit")
            rows[task_id] = " | ".join(row)
    return rows


def test_done_non_educational_tasks_have_audit_coverage() -> None:
    """Every admitted implementation task has a claim/evidence audit row."""
    assert AUDIT.is_file(), "the DOC-001 audit chapter is missing"
    tasks = _ledger_tasks()
    rows = _audit_rows()
    required = {
        task_id
        for task_id, (description, status) in tasks.items()
        if not task_id.startswith("EDU-") and status == "done"
    }
    missing = sorted(required - rows.keys())
    assert not missing, f"done non-EDU tasks missing from audit: {missing}"

    for task_id in required:
        row = rows[task_id]
        assert any(label in row for label in CLAIM_LABELS), (
            f"audit row for {task_id} has no canonical claim label"
        )
        assert len(MARKDOWN_LINK.findall(row)) >= 2, (
            f"audit row for {task_id} lacks linked repository evidence"
        )
        assert re.search(r"invariant|failure|proof", row, re.IGNORECASE), (
            f"audit row for {task_id} lacks an invariant/failure boundary"
        )


def test_partial_active_or_blocked_tasks_are_not_presented_as_final() -> None:
    tasks = _ledger_tasks()
    rows = _audit_rows()
    partial = {
        task_id
        for task_id, (description, status) in tasks.items()
        if not task_id.startswith("EDU-") and status in {"blocked", "in_progress"}
    }
    for task_id in partial:
        assert task_id in rows, f"active/blocked task {task_id} missing from audit"
        row = rows[task_id].lower()
        assert any(
            marker in row
            for marker in ("partial", "blocked", "in progress", "proposed")
        ), f"active/blocked task {task_id} is missing a non-final marker"


def test_handbook_indexes_each_implementation_chapter_and_sources_once() -> None:
    text = HANDBOOK.read_text(encoding="utf-8")
    expected = {
        path.name
        for path in (ROOT / "docs").glob("[0-9][0-9]-*.md")
        if 13 <= int(path.name[:2]) <= 65
    }
    indexed = set(CHAPTER_FILE.findall(text))
    assert indexed == expected, (
        f"handbook index mismatch: missing={expected - indexed}, extra={indexed - expected}"
    )
    for chapter in expected:
        assert text.count(f"]({chapter})") == 1, (
            f"{chapter} must be indexed exactly once"
        )
    assert text.count("](sources.md)") == 1, (
        "source/evidence ledger must be indexed once"
    )


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("**/*.md"))]
    missing: list[str] = []
    for source in markdown_files:
        for raw_target in MARKDOWN_LINK.findall(source.read_text(encoding="utf-8")):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "/", "//")):
                continue
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            target_path = (
                source.parent / target.split("#", 1)[0].split("?", 1)[0]
            ).resolve()
            if (
                not target_path.exists()
                or ROOT not in target_path.parents
                and target_path != ROOT
            ):
                missing.append(f"{source.relative_to(ROOT)} -> {target}")
    assert not missing, "broken local Markdown links:\n" + "\n".join(missing)
