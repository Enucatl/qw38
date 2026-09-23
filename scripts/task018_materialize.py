# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "numpy==2.5.3",
#   "safetensors==0.8.0",
#   "torch==2.9.0",
#   "transformers==5.17.0",
# ]
# ///
"""Materialize the immutable qw38-language-v1 input inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

import transformers

POLICY = Path("docs/architecture/evaluation-policy-v0.md")
P100_SHA = "007757e6c4b340c209aba8a7e159024ce43b0edb237547cdbc3ca51a4853999a"
C92_SHA = "19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23"
TEACHER_CHAT_TEMPLATE_SHA = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)
SYSTEM_PROMPT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)
COMPSEC_INSTRUCTION = (
    "\nAt the end, write exactly one final line in this format and do not "
    "write anything after it:\nAnswer: <line number or comma-separated line numbers>"
)
CHOICE_INSTRUCTION = (
    "\n\nSolve the question. At the end, write exactly one final line in this "
    "format and do not write anything after it:\nAnswer: <letter>"
)
INTEGER_INSTRUCTION = (
    "\nSolve the problem. At the end, write exactly one final line in this "
    "format and do not write anything after it:\nAnswer: <integer>"
)

L12 = [
    ("L01", "Reply with exactly the word blue.", "blue"),
    ("L02", "Rispondi soltanto con la parola rosso.", "rosso"),
    (
        "L03",
        "Text: Mira put the key in the green box. What color is the box? Reply with one word.",
        "green",
    ),
    (
        "L04",
        "Testo: Luca ha tre libri. Quanti libri ha Luca? Rispondi con una cifra.",
        "3",
    ),
    ("L05", "Translate cat into Italian. Reply with one word.", "gatto"),
    ("L06", "Translate cane from Italian into English. Reply with one word.", "dog"),
    (
        "L07",
        "Write the numbers 3, 1, 2 in ascending order, separated by commas with no spaces.",
        "1,2,3",
    ),
    (
        "L08",
        "Reply with a JSON object containing only the key ok with boolean value true. Use no spaces or markdown.",
        '{"ok":true}',
    ),
    ("L09", "What is 17 multiplied by 23? Reply with digits only.", "391"),
    ("L10", "What is the value of 8 + 7 * 2? Reply with digits only.", "22"),
    ("L11", "In Python, what does len([4, 5, 6]) return? Reply with digits only.", "3"),
    ("L12", "In C, what is the integer result of 7 / 2? Reply with digits only.", "3"),
]


def sha256(raw: bytes) -> str:
    """Return the SHA-256 digest of stored bytes."""
    return hashlib.sha256(raw).hexdigest()


def parse_c_string(expression: str) -> str:
    """Decode adjacent C string literals as UTF-8 text."""
    literals = re.findall(r'"(?:\\.|[^"\\])*"', expression)
    if not literals:
        raise ValueError(f"expected a C string literal: {expression[:80]}")
    return "".join(ast.literal_eval(literal) for literal in literals)


def parse_eval_cases(source: str) -> list[dict[str, Any]]:
    """Extract ordered benchmark records from DS4's static C table."""
    start = source.index("static const eval_case eval_cases[] = {")
    end = source.index("\n};", start)
    table = source[start:end]
    records: list[dict[str, Any]] = []
    for block in re.findall(r"\{(.*?)\n    \},", table, re.DOTALL):
        values: dict[str, Any] = {"choices": []}
        pattern = re.compile(
            r"\.(source|id|domain|title|question|answer)\s*=\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+)"
            r"|\.choice\[(\d+)\]\s*=\s*((?:\"(?:\\.|[^\"\\])*\"\s*)+)"
        )
        for match in pattern.finditer(block):
            if match.group(1):
                values[match.group(1)] = parse_c_string(match.group(2))
            else:
                index = int(match.group(3))
                choices = values["choices"]
                while len(choices) <= index:
                    choices.append(None)
                choices[index] = parse_c_string(match.group(4))
        if "id" in values:
            if any(choice is None for choice in values["choices"]):
                raise ValueError(f"non-contiguous choices in {values['id']}")
            records.append(values)
    return records


def question_prompt(case: dict[str, Any]) -> str:
    """Render the DS4 question-body convention selected by EVAL-01."""
    body = case["question"] + "\n"
    if case["choices"]:
        body += "\nChoices:\n"
        body += "".join(
            f"{chr(65 + i)}. {choice}\n" for i, choice in enumerate(case["choices"])
        )
        return body + CHOICE_INSTRUCTION
    if case["source"] == "COMPSEC":
        return body + COMPSEC_INSTRUCTION
    return body + INTEGER_INSTRUCTION


def apply_template(
    tokenizer: Any, user: str, system: str | None
) -> tuple[str, list[int]]:
    """Render and encode one frozen QW38 chat prompt."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return rendered, [int(token) for token in ids]


def retrieval_prompt(seed: int, depth: float, fillers: int) -> str:
    """Construct a deterministic retrieval prompt at a chosen filler count."""
    fact_at = math.floor(depth * fillers)
    rows = ["Read the archive and remember the locker access code.\n"]
    for index in range(fillers + 1):
        if index == fact_at:
            rows.append(
                f"The access code for locker L{seed} is {731204 + 100 * seed}.\n"
            )
        if index < fillers:
            rows.append(f"Record {index:06d}: archive status is unchanged.\n")
    rows.append(
        f"What is the access code for locker L{seed}? Reply with the six digits only."
    )
    return "".join(rows)


def largest_retrieval(
    tokenizer: Any, seed: int, depth: float, horizon: int
) -> tuple[str, str, list[int], int]:
    """Find the greatest filler count whose complete prompt fits a horizon."""
    cache: dict[int, tuple[str, list[int]]] = {}

    def evaluate(n: int) -> tuple[str, list[int]]:
        if n not in cache:
            user = retrieval_prompt(seed, depth, n)
            rendered, ids = apply_template(tokenizer, user, None)
            cache[n] = rendered, ids
        return cache[n]

    low, high = 1, 1
    while len(evaluate(high)[1]) <= horizon:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if len(evaluate(middle)[1]) <= horizon:
            low = middle
        else:
            high = middle
    rendered, ids = evaluate(low)
    if abs(len(ids) - horizon) > 64:
        raise ValueError(f"R fixture {horizon}/{seed}/{depth} is {len(ids)} tokens")
    fact_text = f"The access code for locker L{seed} is {731204 + 100 * seed}.\n"
    # Locate the actual insertion point from the unique fact string in rendered chat text.
    char_position = rendered.index(fact_text)
    prefix_ids = tokenizer(rendered[:char_position], add_special_tokens=False)[
        "input_ids"
    ]
    return rendered, retrieval_prompt(seed, depth, low), ids, len(prefix_ids)


def encode_case(
    tokenizer: Any,
    case_id: str,
    family: str,
    user: str,
    system: str | None,
    **details: Any,
) -> dict[str, Any]:
    """Render a case and retain its exact tokenized prompt and metadata."""
    rendered, ids = apply_template(tokenizer, user, system)
    return {
        "id": case_id,
        "family": family,
        "user_prompt": user,
        "rendered_prompt": rendered,
        "prompt_token_ids": ids,
        "prompt_tokens": len(ids),
        "details": details,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write canonical UTF-8/LF JSONL and return its byte digest."""
    raw = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    path.write_bytes(raw)
    return sha256(raw)


def write_token_files(root: Path, cases: list[dict[str, Any]]) -> None:
    """Write hashed prompt and known retrieval-target arrays and masks."""
    directory = root / "tokens"
    directory.mkdir(parents=True, exist_ok=True)
    target_directory = root / "targets"
    target_directory.mkdir(parents=True, exist_ok=True)
    for case in cases:
        raw = struct.pack(
            f"<{len(case['prompt_token_ids'])}I", *case["prompt_token_ids"]
        )
        path = directory / f"{case['id']}.prompt.u32le"
        path.write_bytes(raw)
        case["prompt_token_file"] = str(path.relative_to(root))
        case["prompt_token_file_bytes"] = len(raw)
        case["prompt_token_sha256"] = sha256(raw)
        del case["prompt_token_ids"]
        target_ids = case.pop("target_token_ids", None)
        if target_ids is not None:
            target_raw = struct.pack(f"<{len(target_ids)}I", *target_ids)
            mask_raw = bytes([1]) * len(target_ids)
            target_path = target_directory / f"{case['id']}.target.u32le"
            mask_path = target_directory / f"{case['id']}.loss-mask.u8"
            target_path.write_bytes(target_raw)
            mask_path.write_bytes(mask_raw)
            case["details"].update(
                {
                    "target_token_file": str(target_path.relative_to(root)),
                    "target_token_count": len(target_ids),
                    "target_token_bytes": len(target_raw),
                    "target_token_sha256": sha256(target_raw),
                    "loss_mask_file": str(mask_path.relative_to(root)),
                    "loss_mask_bytes": len(mask_raw),
                    "loss_mask_sha256": sha256(mask_raw),
                }
            )


def main() -> int:
    """Materialize, validate, and hash all selected EVAL-01 prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ds4-root", type=Path, default=Path("../ds4"))
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=Path("models/Qwen3.8-27B-Q4_K_M.gguf"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prompt_path = args.ds4_root / "gguf-tools/quality-testing/prompts.jsonl"
    prompt_readme_path = args.ds4_root / "gguf-tools/quality-testing/README.md"
    eval_path = args.ds4_root / "ds4_eval.c"
    license_path = args.ds4_root / "LICENSE"
    if not args.teacher_model.is_file():
        raise SystemExit(f"teacher GGUF is missing: {args.teacher_model}")
    prompt_digest = sha256(prompt_path.read_bytes())
    eval_digest = sha256(eval_path.read_bytes())
    if prompt_digest != P100_SHA or eval_digest != C92_SHA:
        raise SystemExit("DS4 source file digest differs from EVAL-01")
    p100_all = [
        json.loads(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
    ]
    p100 = p100_all[:100]
    if [row["id"] for row in p100] != [f"case_{i:03d}" for i in range(100)]:
        raise SystemExit("P100 case inventory is invalid")
    c92 = parse_eval_cases(eval_path.read_text(encoding="utf-8"))
    if len(c92) != 92:
        raise SystemExit(f"C92 case count is {len(c92)}, expected 92")
    for case in c92:
        if case["source"] == "GPQA Diamond (modified)":
            case["source_label_in_ds4"] = case["source"]
            case["modified_question"] = True
            case["source"] = "GPQA Diamond"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    cases: list[dict[str, Any]] = []
    italian = (
        set(range(10, 20))
        | set(range(30, 40))
        | set(range(50, 60))
        | set(range(70, 80))
        | {82, 83, 97, 98, 99}
    )
    code = {5, 15, 21, 31, 43, 53, 60, 63, 70, 73, 81, 92}
    for row in p100:
        number = int(row["id"][-3:])
        slices = ["P100 Italian" if number in italian else "P100 English"]
        if number in code:
            slices.append("P100 Code production")
        cases.append(
            encode_case(
                tokenizer,
                row["id"],
                "P100",
                row["prompt"],
                None,
                slices=slices,
                source_prompt_sha256=prompt_digest,
                cap=256,
            )
        )
    for row in c92:
        cases.append(
            encode_case(
                tokenizer,
                row["id"],
                "C92",
                question_prompt(row),
                SYSTEM_PROMPT,
                source=row["source"],
                source_label_in_ds4=row.get("source_label_in_ds4", row["source"]),
                modified_question=row.get("modified_question", False),
                domain=row["domain"],
                title=row["title"],
                choices=row["choices"],
                answer=row["answer"],
                cap=2048,
            )
        )
    for case_id, user, answer in L12:
        cases.append(
            encode_case(tokenizer, case_id, "L12", user, None, expected=answer, cap=32)
        )
    retrieval: list[dict[str, Any]] = []
    for horizon in (512, 4096, 32768):
        for seed in (0, 1):
            for depth in (0.1, 0.5, 0.9):
                rendered, user, ids, fact_pos = largest_retrieval(
                    tokenizer, seed, depth, horizon
                )
                retrieval_case = {
                    "id": f"R-{horizon}-s{seed}-d{depth:.1f}",
                    "family": "R",
                    "user_prompt": user,
                    "rendered_prompt": rendered,
                    "prompt_token_ids": ids,
                    "prompt_tokens": len(ids),
                    "target_token_ids": tokenizer.encode(
                        str(731204 + 100 * seed), add_special_tokens=False
                    ),
                    "details": {
                        "horizon": horizon,
                        "seed": seed,
                        "depth_label": depth,
                        "fact_token_position": fact_pos,
                        "answer": str(731204 + 100 * seed),
                        "cap": 32,
                    },
                }
                retrieval.append(retrieval_case)
    cases.extend(retrieval)

    # Persist the source records separately so every prompt/key can be audited.
    args.output.mkdir(parents=True, exist_ok=True)
    vendored = args.output / "source"
    vendored.mkdir(exist_ok=True)
    (vendored / "p100.jsonl").write_bytes(prompt_path.read_bytes())
    (vendored / "p100.README.md").write_bytes(prompt_readme_path.read_bytes())
    (vendored / "ds4_eval.c").write_bytes(eval_path.read_bytes())
    (vendored / "ds4.LICENSE").write_bytes(license_path.read_bytes())
    write_token_files(args.output, cases)
    prompt_jsonl_hash = write_jsonl(args.output / "prompts.jsonl", cases)

    tokenizer_files = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
    ):
        path = args.checkpoint / name
        if path.exists():
            tokenizer_files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
    ids = tokenizer.get_added_vocab()
    generation_config = json.loads(
        (args.checkpoint / "generation_config.json").read_text()
    )
    eos = generation_config["eos_token_id"]
    manifest = {
        "suite": "qw38-language-v1",
        "fixture_version": "1.0.0",
        "policy_path": str(POLICY),
        "policy_sha256": sha256(POLICY.read_bytes()),
        "ds4_revision": "c238077a87186381bf626cc531bccffe1fef79e7",
        "sources": {
            "p100": {
                "path": str(prompt_path),
                "vendored_path": "source/p100.jsonl",
                "sha256": prompt_digest,
                "license": "MIT (DS4 repository LICENSE)",
                "license_file_sha256": sha256(license_path.read_bytes()),
                "readme_sha256": sha256(prompt_readme_path.read_bytes()),
                "attribution": "DS4 curated prompts from revision c238077a87186381bf626cc531bccffe1fef79e7; the prompt JSONL and README provide no upstream prompt-dataset citation.",
            },
            "c92": {
                "path": str(eval_path),
                "vendored_path": "source/ds4_eval.c",
                "sha256": eval_digest,
                "attribution": "DS4 eval_cases; source header credits GPQA Diamond (CC BY 4.0), SuperGPQA (ODC-BY), AIME 2025 mirror (MIT), and COMPSEC reductions derived from public CVE writeups.",
            },
        },
        "tokenizer": {
            "checkpoint": str(args.checkpoint),
            "files": tokenizer_files,
            "added_vocab_sha256": sha256(json.dumps(ids, sort_keys=True).encode()),
            "special_token_ids": {
                "bos": tokenizer.bos_token_id,
                "eos": tokenizer.eos_token_id,
                "pad": tokenizer.pad_token_id,
            },
            "generation_eos_ids": eos,
            "template_arguments": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
        "source_teacher": {
            "path": str(args.teacher_model),
            "file_size_bytes": args.teacher_model.stat().st_size,
            "gguf_name": "Qwen3.8-27B",
            "gguf_architecture": "qwen35",
            "gguf_file_type": 15,
            "quantization": "Q4_K_M",
            "tokenizer_model": "gpt2",
            "tokenizer_template_sha256": TEACHER_CHAT_TEMPLATE_SHA,
            "bos_token_id": 248044,
            "eos_token_ids": [248046],
            "stop_token_ids": [248044, 248046, 248063, 248064, 248065],
            "generation": {
                "backend": "CUDA",
                "gpu_layers": "all",
                "endpoint": "llama-server /completion",
                "tokenizer_endpoint": "/tokenize",
                "greedy": True,
                "temperature": 0.0,
                "samplers": ["temperature"],
                "max_tokens": 24,
                "stop_token_ids": [248044, 248046, 248063, 248064, 248065],
                "ignore_eos": False,
                "add_special": False,
                "parse_special": True,
                "chat_template_applied_by_server": False,
            },
        },
        "inventory": {
            "P100": 100,
            "C92": 92,
            "L12": 12,
            "R": 18,
            "R_core": 12,
            "R_deferred_TASK_022": 6,
            "cases_total": len(cases),
        },
        "files": {
            "prompts.jsonl": {
                "bytes": (args.output / "prompts.jsonl").stat().st_size,
                "sha256": prompt_jsonl_hash,
            }
        },
        "scoring_implementation": {
            "path": "scripts/task018_scoring.py",
            "sha256": sha256(Path("scripts/task018_scoring.py").read_bytes()),
            "grader_version": "qw38-language-v1-eval01",
        },
        "materializer_implementation": {
            "path": "scripts/task018_materialize.py",
            "sha256": sha256(Path(__file__).read_bytes()),
        },
        "metrics_implementation": {
            "path": "scripts/task018_metrics.py",
            "sha256": sha256(Path("scripts/task018_metrics.py").read_bytes()),
            "resampler": "sha256-counter-v1",
        },
        "reference_capture": {
            "status": "NOT_CAPTURED",
            "required_before_comparison": True,
        },
        "coverage": "fixtures_materialized; no model comparison has been run",
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    (args.output / "manifest.json").write_bytes(manifest_raw)
    (args.output / "manifest.sha256").write_text(
        sha256(manifest_raw) + "  manifest.json\n"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "prompts_sha256": prompt_jsonl_hash,
                "manifest_sha256": sha256(manifest_raw),
                "reference_capture": "NOT_CAPTURED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
