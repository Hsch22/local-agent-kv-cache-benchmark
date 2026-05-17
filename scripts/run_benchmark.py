#!/usr/bin/env python3
"""Run one JSONL workload against a local OpenAI-compatible llama-server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from openai import OpenAI
from tqdm import tqdm


CSV_FIELDS = [
    "run_id",
    "timestamp",
    "machine",
    "model",
    "strategy",
    "cache_prompt",
    "cache_reuse",
    "workload",
    "layout",
    "prompt_len_target",
    "turn_id",
    "repeat_id",
    "cache_state",
    "max_tokens",
    "temperature",
    "top_p",
    "ttft_s",
    "total_latency_s",
    "decode_latency_s",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_s",
    "server_pid",
    "server_rss_mb",
    "server_cpu_pct",
    "error",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def delta_text(event: Any) -> str:
    try:
        choice = event.choices[0]
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None)
        return content or ""
    except Exception:
        return ""


def usage_from_event(event: Any) -> tuple[int | None, int | None]:
    usage = getattr(event, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def server_stats(pid: int | None) -> tuple[float | None, float | None]:
    if not pid:
        return None, None
    try:
        process = psutil.Process(pid)
        rss_mb = process.memory_info().rss / 1024 / 1024
        cpu_pct = process.cpu_percent(interval=None)
        return rss_mb, cpu_pct
    except psutil.Error:
        return None, None


def count_output_tokens_rough(text: str) -> int:
    return max(1, len(text.split()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Workload JSONL path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "local"))
    parser.add_argument("--strategy", default="manual")
    parser.add_argument("--cache-prompt", choices=["true", "false"], default="true")
    parser.add_argument("--cache-reuse", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_path)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    run_id = f"{input_path.stem}-{int(time.time())}"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for repeat_id in range(1, args.repeat + 1):
            iterator = tqdm(records, desc=f"repeat {repeat_id}", unit="req")
            for record in iterator:
                started = time.perf_counter()
                first_token = None
                output_text = []
                prompt_tokens = None
                completion_tokens = None
                error = ""

                try:
                    stream = client.chat.completions.create(
                        model=args.model,
                        messages=record["messages"],
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    for event in stream:
                        text = delta_text(event)
                        usage_prompt, usage_completion = usage_from_event(event)
                        prompt_tokens = usage_prompt if usage_prompt is not None else prompt_tokens
                        completion_tokens = (
                            usage_completion if usage_completion is not None else completion_tokens
                        )
                        if text:
                            if first_token is None:
                                first_token = time.perf_counter()
                            output_text.append(text)
                except Exception as exc:
                    error = repr(exc)

                ended = time.perf_counter()
                ttft_s = (first_token - started) if first_token else None
                total_latency_s = ended - started
                decode_latency_s = (
                    total_latency_s - ttft_s if ttft_s is not None else None
                )
                joined = "".join(output_text)
                if completion_tokens is None and joined:
                    completion_tokens = count_output_tokens_rough(joined)
                tokens_per_s = (
                    completion_tokens / decode_latency_s
                    if completion_tokens and decode_latency_s and decode_latency_s > 0
                    else None
                )
                rss_mb, cpu_pct = server_stats(args.server_pid)

                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "machine": "MacBookPro-M5-32GB",
                    "model": args.model,
                    "strategy": args.strategy,
                    "cache_prompt": args.cache_prompt,
                    "cache_reuse": args.cache_reuse,
                    "workload": record.get("workload"),
                    "layout": record.get("layout"),
                    "prompt_len_target": record.get("prompt_len_target"),
                    "turn_id": record.get("turn_id"),
                    "repeat_id": repeat_id,
                    "cache_state": "cold" if record.get("turn_id") == 1 else "warm",
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "ttft_s": ttft_s,
                    "total_latency_s": total_latency_s,
                    "decode_latency_s": decode_latency_s,
                    "prompt_tokens": prompt_tokens or record.get("approx_prompt_tokens"),
                    "completion_tokens": completion_tokens,
                    "tokens_per_s": tokens_per_s,
                    "server_pid": args.server_pid,
                    "server_rss_mb": rss_mb,
                    "server_cpu_pct": cpu_pct,
                    "error": error,
                }
                writer.writerow(row)
                handle.flush()

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
