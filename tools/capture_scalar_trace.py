"""Capture one filtered real scalar tap into a validated v1 trace bundle."""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from tools.qw38_trace import (
        ArtifactIdentity,
        SessionSnapshot,
        TraceError,
        TraceTensor,
        read_trace_bundle,
        write_trace_bundle,
    )
else:
    from qw38_trace import (  # type: ignore[import-not-found]
        ArtifactIdentity,
        SessionSnapshot,
        TraceError,
        TraceTensor,
        read_trace_bundle,
        write_trace_bundle,
    )

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "0669b98607d47046c7c2b3f801011d54a08cfccf"
MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
    return result.stdout.strip() + ("+dirty" if dirty else "")


def _fields(stdout: str) -> dict[str, str]:
    try:
        return dict(line.split("=", 1) for line in stdout.splitlines())
    except ValueError as error:
        raise TraceError("native scalar trace metadata is malformed") from error


def capture_scalar_trace(
    model: Path,
    executable: Path,
    destination: Path,
    *,
    token: int,
    layer: int | None,
    tap: str,
) -> Path:
    """Run one real token and wrap its selected native tensor in trace v1."""

    if _sha256(model) != MODEL_SHA256:
        raise TraceError("model does not match the pinned trace identity")
    layer_text = "global" if layer is None else str(layer)
    with tempfile.TemporaryDirectory(prefix="qw38-scalar-trace-") as temporary:
        raw_path = Path(temporary) / "tap.f32le.bin"
        result = subprocess.run(
            [
                str(executable),
                "--capture-real-scalar-trace",
                str(model),
                str(token),
                layer_text,
                tap,
                str(raw_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TraceError(result.stderr.strip() or "native scalar trace failed")
        metadata = _fields(result.stdout)
        required = {
            "tap_name",
            "tap_layer",
            "shape",
            "count",
            "token",
            "position",
            "frontier_before",
            "frontier_after",
        }
        if not required.issubset(metadata):
            raise TraceError("native scalar trace metadata is incomplete")
        shape = tuple(int(value) for value in metadata["shape"].split(","))
        count = int(metadata["count"])
        raw = raw_path.read_bytes()
        if len(raw) != count * 4:
            raise TraceError("native scalar trace byte count does not match")
        values = struct.unpack(f"<{count}f", raw)
        state_names = (
            "gdn_convolution",
            "gdn_recurrent",
            "attention_key",
            "attention_value",
        )
        before = {name: metadata[f"state_before_{name}"] for name in state_names}
        after = {name: metadata[f"state_after_{name}"] for name in state_names}
        tensor_layer = int(metadata["tap_layer"])
        manifest = write_trace_bundle(
            destination,
            model=ArtifactIdentity(model.name, MODEL_REVISION, MODEL_SHA256),
            tool=ArtifactIdentity(executable.name, _revision(), _sha256(executable)),
            prompt=b"",
            token_ids=[int(metadata["token"])],
            positions=[int(metadata["position"])],
            session_before=SessionSnapshot(int(metadata["frontier_before"]), before),
            session_after=SessionSnapshot(int(metadata["frontier_after"]), after),
            tensors=[
                TraceTensor(
                    name=metadata["tap_name"],
                    role=metadata["tap_name"],
                    layer=None if tensor_layer == 64 else tensor_layer,
                    shape=shape,
                    values=values,
                )
            ],
        )
    read_trace_bundle(destination)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--executable", type=Path, default=ROOT / "build/qw38-eval-diagnostic"
    )
    parser.add_argument("--token", type=int, default=42)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--tap", required=True)
    arguments = parser.parse_args()
    capture_scalar_trace(
        arguments.model,
        arguments.executable,
        arguments.destination,
        token=arguments.token,
        layer=arguments.layer,
        tap=arguments.tap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
