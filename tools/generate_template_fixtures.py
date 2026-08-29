from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import Environment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models" / "tokenizer" / "tokenizer_config.json"
TOKENIZER = ROOT / "models" / "tokenizer" / "tokenizer.json"
OUTPUT = ROOT / "fixtures" / "template_authority.json"

SUCCESS_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "user_no_thinking",
        "messages": [{"role": "user", "content": "Hello"}],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
    },
    {
        "name": "user_default_xhigh",
        "messages": [{"role": "user", "content": "Explain quartz."}],
        "options": {"add_generation_prompt": True},
    },
    {
        "name": "system_low",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Why a watch?"},
        ],
        "options": {
            "add_generation_prompt": True,
            "enable_thinking": True,
            "reasoning_effort": "low",
        },
    },
    {
        "name": "assistant_history",
        "messages": [
            {"role": "user", "content": "2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Two plus two is four.",
                "content": "4",
            },
            {"role": "user", "content": "And plus 3?"},
        ],
        "options": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    },
    {
        "name": "tools_and_result",
        "messages": [
            {"role": "user", "content": "Weather in Bern?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should check.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Bern", "unit": "C"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature":18}'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": True},
    },
)

ERROR_CASES: tuple[dict[str, Any], ...] = (
    {"name": "no_messages", "messages": [], "options": {}},
    {
        "name": "late_system",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Too late"},
        ],
        "options": {},
    },
    {
        "name": "invalid_reasoning_effort",
        "messages": [{"role": "user", "content": "Hi"}],
        "options": {"reasoning_effort": "maximum"},
    },
)

POLICY_ERRORS: tuple[dict[str, Any], ...] = (
    {
        "name": "image_rejected_in_v1",
        "messages": [{"role": "user", "content": [{"type": "image", "url": "x"}]}],
        "error": "vision content is unsupported in v1",
        "owner": "quartz_v1_policy",
    },
)

POLICY_SUCCESSES: tuple[dict[str, Any], ...] = (
    {
        "name": "leading_developer_mapped_to_system",
        "messages": [
            {"role": "developer", "content": "Follow API policy."},
            {"role": "user", "content": "Hi"},
        ],
        "authority_messages": [
            {"role": "system", "content": "Follow API policy."},
            {"role": "user", "content": "Hi"},
        ],
        "options": {"add_generation_prompt": True, "enable_thinking": False},
        "owner": "quartz_v1_policy",
    },
)


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    template_text = config["chat_template"]
    environment = Environment()
    environment.globals["raise_exception"] = fail
    template = environment.from_string(template_text)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    successes: list[dict[str, Any]] = []
    for case in SUCCESS_CASES:
        arguments = {
            "messages": case["messages"],
            "tools": case.get("tools"),
            "add_vision_id": False,
            **case["options"],
        }
        rendered = template.render(**arguments)
        successes.append(
            {
                **case,
                "rendered_utf8_hex": rendered.encode().hex(),
                "ids": tokenizer.encode(rendered, add_special_tokens=False).ids,
            }
        )

    errors: list[dict[str, Any]] = []
    for case in ERROR_CASES:
        try:
            template.render(
                messages=case["messages"],
                tools=case.get("tools"),
                add_vision_id=False,
                **case["options"],
            )
        except (ValueError, jinja2.TemplateError) as error:
            errors.append({**case, "error": str(error)})
        else:
            raise SystemExit(f"template unexpectedly accepted {case['name']}")

    policy_successes: list[dict[str, Any]] = []
    for case in POLICY_SUCCESSES:
        rendered = template.render(
            messages=case["authority_messages"],
            tools=None,
            add_vision_id=False,
            **case["options"],
        )
        policy_successes.append(
            {
                **case,
                "rendered_utf8_hex": rendered.encode().hex(),
                "ids": tokenizer.encode(rendered, add_special_tokens=False).ids,
            }
        )

    document = {
        "schema_version": 1,
        "authority": {
            "repository": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "tokenizer_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "chat_template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
            "jinja2_version": jinja2.__version__,
        },
        "successes": successes,
        "errors": errors,
        "policy_successes": policy_successes,
        "policy_errors": POLICY_ERRORS,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
