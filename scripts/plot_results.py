#!/usr/bin/env python3
"""Summarize raw benchmark CSV files and write figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_raw(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no CSV files found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
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
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame[(frame["error"].fillna("") == "") & (frame["cache_state"] == "warm")].copy()
    group_cols = ["strategy", "workload", "layout", "prompt_len_target"]
    summary = (
        usable.groupby(group_cols)
        .agg(
            median_ttft_s=("ttft_s", "median"),
            p95_ttft_s=("ttft_s", lambda values: values.quantile(0.95)),
            median_total_latency_s=("total_latency_s", "median"),
            median_tokens_per_s=("tokens_per_s", "median"),
            peak_server_rss_mb=("server_rss_mb", "max"),
            samples=("ttft_s", "count"),
        )
        .reset_index()
    )
    baseline = summary[summary["strategy"] == "no_cache"][
        ["workload", "prompt_len_target", "median_ttft_s"]
    ].rename(columns={"median_ttft_s": "baseline_median_ttft_s"})
    summary = summary.merge(baseline, on=["workload", "prompt_len_target"], how="left")
    summary["ttft_speedup"] = summary["baseline_median_ttft_s"] / summary["median_ttft_s"]
    return summary


def save_figures(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=summary,
        x="prompt_len_target",
        y="median_ttft_s",
        hue="strategy",
        style="workload",
        markers=True,
    )
    plt.title("Warm TTFT by prompt length")
    plt.xlabel("Prompt length target")
    plt.ylabel("Median warm TTFT (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "ttft_by_length.png", dpi=160)
    plt.close()

    if "ttft_speedup" in summary:
        plt.figure(figsize=(9, 5))
        sns.lineplot(
            data=summary,
            x="prompt_len_target",
            y="ttft_speedup",
            hue="strategy",
            style="workload",
            markers=True,
        )
        plt.axhline(1.0, color="black", linewidth=1)
        plt.title("Warm TTFT speedup vs no cache")
        plt.xlabel("Prompt length target")
        plt.ylabel("Speedup")
        plt.tight_layout()
        plt.savefig(output_dir / "speedup_by_length.png", dpi=160)
        plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="layout", y="median_ttft_s", hue="strategy")
    plt.title("Layout sensitivity")
    plt.xlabel("Prompt layout")
    plt.ylabel("Median warm TTFT (s)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "layout_sensitivity.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/raw")
    parser.add_argument("--summary-dir", default="results/summary")
    parser.add_argument("--figures-dir", default="results/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame = numeric_columns(read_raw(Path(args.input_dir)))
    summary = summarize(frame)
    summary_path = summary_dir / "formal_all_raw_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_figures(summary, Path(args.figures_dir))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
