#!/usr/bin/env python3
"""Start llama-server sequentially for benchmark matrices and run workloads."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


STRATEGIES = {
    "S0": {"name": "no_cache", "cache_prompt": "false", "server_args": ["--no-cache-prompt"]},
    "S1": {"name": "default_cache", "cache_prompt": "true", "server_args": ["--cache-prompt"]},
    "S2": {"name": "stable_prefix", "cache_prompt": "true", "server_args": ["--cache-prompt"]},
}

DEFAULT_LAYOUT_BY_STRATEGY = {
    "S0": "stable_prefix",
    "S1": "front_volatile",
    "S2": "stable_prefix",
}

LAYOUTS = ("front_volatile", "middle_volatile", "end_volatile", "stable_prefix")


def wait_ready(base_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        for suffix in ("/v1/models", "/health"):
            try:
                response = requests.get(base_url.rstrip("/") + suffix, timeout=2)
                if response.status_code < 500:
                    return
            except requests.RequestException as exc:
                last_error = exc
        time.sleep(1)
    raise RuntimeError(f"server was not ready within {timeout_s}s: {last_error}")


def start_server(args: argparse.Namespace, strategy: str, ctx_size: int) -> subprocess.Popen:
    strategy_spec = STRATEGIES[strategy]
    log_path = Path(args.log_dir) / f"llama_server_{strategy}_{ctx_size}_{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    command = [
        args.llama_server,
        "-m",
        args.model_path,
        "-c",
        str(ctx_size),
        "-ngl",
        str(args.gpu_layers),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--parallel",
        "1",
        "--metrics",
        *strategy_spec["server_args"],
    ]
    if args.cache_reuse and strategy != "S0":
        command.extend(["--cache-reuse", str(args.cache_reuse)])

    print("starting:", " ".join(command))
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_benchmark(
    args: argparse.Namespace,
    strategy: str,
    workload_file: Path,
    process: subprocess.Popen,
    suite_repeat_id: int,
) -> None:
    strategy_spec = STRATEGIES[strategy]
    prefix = f"{args.run_label}_" if args.run_label else ""
    reuse_suffix = f"_reuse{args.cache_reuse}" if args.cache_reuse != "" else ""
    output_name = (
        f"{prefix}{workload_file.stem}_{strategy_spec['name']}"
        f"{reuse_suffix}_r{suite_repeat_id:02d}.csv"
    )
    output_path = Path(args.results_dir) / output_name
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing result: {output_path}")
    command = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--input",
        str(workload_file),
        "--output",
        str(output_path),
        "--base-url",
        f"http://{args.host}:{args.port}/v1",
        "--model",
        args.model_name,
        "--strategy",
        strategy_spec["name"],
        "--cache-prompt",
        strategy_spec["cache_prompt"],
        "--cache-reuse",
        str(args.cache_reuse or ""),
        "--repeat",
        "1",
        "--max-tokens",
        str(args.max_tokens),
        "--server-pid",
        str(process.pid),
    ]
    subprocess.run(command, check=True)


def prompt_length_to_ctx(length_label: str) -> int:
    mapping = {"2k": 4096, "4k": 8192, "8k": 16384, "16k": 32768}
    return mapping[length_label]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "local"))
    parser.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER", "llama-server"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--workload-dir", default="workloads")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Number of server-level repeats. Each repeat restarts llama-server.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cache-reuse", default="")
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--workloads", nargs="+", choices=["coding", "rag"], default=None)
    parser.add_argument("--lengths", nargs="+", choices=["2k", "4k", "8k", "16k"], default=None)
    parser.add_argument("--strategies", nargs="+", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_matrix(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.mode == "smoke":
        return [
            ("S0", "coding_2k_stable_prefix.jsonl", "2k"),
            ("S2", "coding_2k_stable_prefix.jsonl", "2k"),
        ]

    workloads = args.workloads or ["coding"]
    lengths = args.lengths or ["4k", "8k", "16k"]
    strategies = args.strategies or ["S0", "S1", "S2"]

    matrix = []
    for workload in workloads:
        for length in lengths:
            for strategy in strategies:
                layouts = args.layouts or [DEFAULT_LAYOUT_BY_STRATEGY[strategy]]
                for layout in layouts:
                    matrix.append((strategy, f"{workload}_{length}_{layout}.jsonl", length))
    return matrix


def main() -> int:
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path is required or MODEL_PATH must be set")
    if not Path(args.model_path).exists():
        raise SystemExit(f"model file does not exist: {args.model_path}")

    matrix = build_matrix(args)
    print("planned sequential configs:")
    for index, (strategy, workload_name, length_label) in enumerate(matrix, start=1):
        print(f"  {index:02d}. {strategy} {length_label} {workload_name}")

    for strategy, workload_name, length_label in matrix:
        workload_file = Path(args.workload_dir) / workload_name
        if not workload_file.exists():
            raise SystemExit(f"missing workload: {workload_file}")
        for suite_repeat_id in range(1, args.repeat + 1):
            print(
                f"running {strategy} {workload_name} "
                f"server-repeat {suite_repeat_id}/{args.repeat}"
            )
            process = start_server(args, strategy, prompt_length_to_ctx(length_label))
            try:
                wait_ready(f"http://{args.host}:{args.port}")
                run_benchmark(args, strategy, workload_file, process, suite_repeat_id)
            finally:
                stop_server(process)
                time.sleep(args.cooldown_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
