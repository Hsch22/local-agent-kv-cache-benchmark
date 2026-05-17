#!/usr/bin/env python3
"""Summarize selected raw benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "cache_reuse",
    "prompt_len_target",
    "turn_id",
    "repeat_id",
    "ttft_s",
    "total_latency_s",
    "decode_latency_s",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_s",
    "server_rss_mb",
]


def read_inputs(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        for path in sorted(Path().glob(pattern)):
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        joined = ", ".join(patterns)
        raise SystemExit(f"no CSV files matched: {joined}")
    return pd.concat(frames, ignore_index=True)


def numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[(frame["error"].fillna("") == "") & (frame["cache_state"] == "warm")].copy()
    group_cols = [
        "workload",
        "prompt_len_target",
        "layout",
        "strategy",
        "cache_prompt",
        "cache_reuse",
    ]
    group_cols = [column for column in group_cols if column in usable]
    summary = (
        usable.groupby(group_cols, dropna=False)
        .agg(
            runs=("source_file", "nunique"),
            samples=("ttft_s", "count"),
            median_ttft_s=("ttft_s", "median"),
            p95_ttft_s=("ttft_s", lambda values: values.quantile(0.95)),
            median_total_latency_s=("total_latency_s", "median"),
            median_decode_latency_s=("decode_latency_s", "median"),
            median_tokens_per_s=("tokens_per_s", "median"),
            median_prompt_tokens=("prompt_tokens", "median"),
            median_completion_tokens=("completion_tokens", "median"),
            peak_server_rss_mb=("server_rss_mb", "max"),
        )
        .reset_index()
        .sort_values(group_cols)
    )

    if "layout" in summary:
        layout_base = summary[summary["layout"] == "front_volatile"][
            ["workload", "prompt_len_target", "median_ttft_s"]
        ].rename(columns={"median_ttft_s": "front_volatile_median_ttft_s"})
        summary = summary.merge(layout_base, on=["workload", "prompt_len_target"], how="left")
        summary["ttft_speedup_vs_front_volatile"] = (
            summary["front_volatile_median_ttft_s"] / summary["median_ttft_s"]
        )

    if "cache_reuse" in summary:
        reuse_base = summary[summary["cache_reuse"].fillna(-1) == 0][
            ["workload", "prompt_len_target", "layout", "median_ttft_s"]
        ].rename(columns={"median_ttft_s": "reuse0_median_ttft_s"})
        summary = summary.merge(
            reuse_base,
            on=["workload", "prompt_len_target", "layout"],
            how="left",
        )
        summary["ttft_speedup_vs_reuse0"] = (
            summary["reuse0_median_ttft_s"] / summary["median_ttft_s"]
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="CSV glob patterns to summarize")
    parser.add_argument("--output", required=True, help="Output summary CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(numeric_columns(read_inputs(args.patterns)))
    summary.to_csv(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
