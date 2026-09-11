"""Task-owned deterministic Qwen continuation fixtures. Not ds4 DeepSeek/GLM."""

from __future__ import annotations

from typing import Any

from tools.opt058_quality_baseline import (
    NO_THINKING_SUFFIX,
    render_no_thinking_user_turn,
)

# DeepSeek/GLM official continuation directories from ds4. Methodology only;
# these are not Qwen continuations and must not be scored as such.
DS4_DATASETS_NOT_APPLICABLE: tuple[str, ...] = (
    "data/glm52-openrouter-100",
    "data/flash",
    "data/pro",
    "data/pro-0813",
)

# Tiny synthetic Qwen cases owned by OPT-083. Token IDs are frozen for
# framework identity; OPT-084 scores the shipping model on the real spans.
QWEN_CONTINUATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen_capitals_paris",
        "user": "Name the capital of France in one word.",
        "continuation_text": "Paris",
        "context": (151644, 872, 198, 5646, 279, 6722, 315, 9625, 304, 825, 3405),
        "targets": (21567, 13),
        "source": "opt083_deterministic_qwen",
        "remote": False,
    },
    {
        "id": "qwen_two_plus_two",
        "user": "What is 2 + 2? Reply with the integer only.",
        "continuation_text": "4",
        "context": (151644, 872, 198, 3838, 374, 220, 17, 488, 220, 17, 30),
        "targets": (19,),
        "source": "opt083_deterministic_qwen",
        "remote": False,
    },
)


def qwen_case_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in QWEN_CONTINUATIONS:
        rendered = render_no_thinking_user_turn(str(case["user"]))
        rows.append(
            {
                **case,
                "rendered": rendered,
                "assistant_answer_boundary": NO_THINKING_SUFFIX,
                "chat_template": "no_thinking",
                "enable_thinking": False,
                "scoring": "teacher_forced_nll",
                "logit_masking": False,
                "ds4_dataset": False,
            }
        )
    return rows
