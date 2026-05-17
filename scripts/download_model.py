#!/usr/bin/env python3
"""Download the benchmark GGUF model into the project models directory."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-cache" / "huggingface"))
os.environ.setdefault("HF_XET_CACHE", str(PROJECT_ROOT / ".hf-cache" / "xet"))

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
DEFAULT_FILENAME = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
DEFAULT_SHA256 = "724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--local-dir", default="models")
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip checksum verification after download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=local_dir,
        )
    ).resolve()

    if not args.skip_sha256 and args.expected_sha256:
        actual = sha256_file(path)
        if actual.lower() != args.expected_sha256.lower():
            raise SystemExit(
                "SHA256 mismatch for "
                f"{path}\nexpected: {args.expected_sha256}\nactual:   {actual}"
            )

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
