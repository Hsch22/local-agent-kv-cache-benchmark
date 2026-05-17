#!/usr/bin/env python3
"""Plot follow-up experiment summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_layout(summary_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(summary_path)
    order = ["front_volatile", "middle_volatile", "end_volatile", "stable_prefix"]
    plt.figure(figsize=(8, 4.8))
    sns.barplot(
        data=frame,
        x="layout",
        y="median_ttft_s",
        hue="workload",
        order=order,
    )
    plt.title("8K layout sensitivity")
    plt.xlabel("Layout")
    plt.ylabel("Median warm TTFT (s)")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "formal_layout8k_sensitivity.png", dpi=160)
    plt.close()


def plot_cache_reuse(summary_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(summary_path).sort_values("cache_reuse")
    frame["series"] = (
        frame["workload"].astype(str)
        + " "
        + (frame["prompt_len_target"].astype(int) // 1024).astype(str)
        + "K"
    )
    plt.figure(figsize=(8.4, 4.8))
    sns.lineplot(
        data=frame,
        x="cache_reuse",
        y="median_ttft_s",
        hue="series",
        marker="o",
    )
    plt.title("Stable-prefix cache-reuse sweep")
    plt.xlabel("--cache-reuse")
    plt.ylabel("Median warm TTFT (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "formal_cache_reuse_sweep.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-summary",
        default="results/summary/formal_layout_coding_rag_8k_summary.csv",
    )
    parser.add_argument(
        "--cache-reuse-summary",
        default="results/summary/formal_cache_reuse_coding_rag_8k_16k_summary.csv",
    )
    parser.add_argument("--figures-dir", default="results/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_layout(Path(args.layout_summary), output_dir)
    plot_cache_reuse(Path(args.cache_reuse_summary), output_dir)
    print(output_dir / "formal_layout8k_sensitivity.png")
    print(output_dir / "formal_cache_reuse_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
