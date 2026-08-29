from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "models" / "tokenizer"
OUTPUT = ROOT / "fixtures" / "tokenizer_authority.json"

CASES: tuple[tuple[str, str], ...] = (
    ("empty", ""),
    ("ascii", "hello world"),
    ("contractions", "I'm sure we've tested Qwen's tokenizer."),
    ("spaces", "  leading   middle  trailing  "),
    ("newlines", "first\nsecond\r\n\nlast"),
    ("punctuation", "a+b == c; // comment?!"),
    ("nfc", "café Ångström"),
    ("nfd_input", "cafe\u0301 A\u030angstro\u0308m"),
    ("cjk", "量子手表每秒滴答。"),
    ("mixed_unicode", "مرحبا नमस्ते 🕰️ café"),
    ("code", "def tick(x: int) -> int:\n    return x + 1\n"),
    ("special", "<|im_start|>user\nhello<|im_end|>\n"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    pins = json.loads((ROOT / "pins" / "artifacts.lock.json").read_text())
    tokenizer_pin = pins["tokenizer"]
    for name, expected in tokenizer_pin["files"].items():
        path = ASSETS / name
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise SystemExit(f"tokenizer asset identity mismatch: {name}")

    tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
    fixtures: list[dict[str, object]] = []
    for name, text in CASES:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        fixtures.append(
            {
                "name": name,
                "utf8_hex": text.encode().hex(),
                "ids": encoded.ids,
                "tokens": encoded.tokens,
                "decoded_utf8_hex": tokenizer.decode(encoded.ids).encode().hex(),
            }
        )

    document = {
        "schema_version": 1,
        "authority": {
            "repository": tokenizer_pin["repository"],
            "revision": tokenizer_pin["revision"],
            "tokenizer_json_sha256": tokenizer_pin["files"]["tokenizer.json"]["sha256"],
            "tokenizers_version": tokenizers.__version__,
        },
        "normalizer": tokenizer_pin["normalizer"],
        "cases": fixtures,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
