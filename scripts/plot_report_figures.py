#!/usr/bin/env python3
"""Generate publication-quality benchmark figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


FIG_BG = "#FFFFFF"
AX_BG = "#FFFFFF"
GRID = "#DDE5EF"
TEXT = "#18212F"
MUTED = "#64748B"
SPINE = "#CBD5E1"

STRATEGY_LABELS = {
    "no_cache": "S0 no-cache",
    "default_cache": "S1 default-cache",
    "stable_prefix": "S2 stable-prefix",
}
STRATEGY_COLORS = {
    "no_cache": "#334155",
    "default_cache": "#D97706",
    "stable_prefix": "#008E7D",
}
STRATEGY_MARKERS = {
    "no_cache": "o",
    "default_cache": "s",
    "stable_prefix": "D",
}

WORKLOAD_LABELS = {
    "coding": "Coding agent",
    "rag": "RAG agent",
}

LAYOUT_LABELS = {
    "front_volatile": "Front volatile",
    "middle_volatile": "Middle volatile",
    "end_volatile": "End volatile",
    "stable_prefix": "Stable prefix",
}
LAYOUT_COLORS = {
    "front_volatile": "#D65A31",
    "middle_volatile": "#D99A21",
    "end_volatile": "#008E7D",
    "stable_prefix": "#2563EB",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_BG,
            "axes.facecolor": AX_BG,
            "savefig.facecolor": FIG_BG,
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.labelcolor": TEXT,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 0.95,
            "legend.frameon": False,
            "legend.fontsize": 10.8,
            "legend.handlelength": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def seconds_formatter(value: float, _pos: int) -> str:
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def speedup_formatter(value: float, _pos: int) -> str:
    return f"{value:g}x"


def polish_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=4.5, width=0.9, color=SPINE)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)


def read_main_summary(summary_dir: Path) -> pd.DataFrame:
    paths = [
        summary_dir / "formal_main_coding_rag_2k_summary.csv",
        summary_dir / "formal_main_coding_rag_4k_8k_16k_summary.csv",
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["prompt_len_target"] = pd.to_numeric(frame["prompt_len_target"], errors="coerce")
    frame["length_k"] = (frame["prompt_len_target"] / 1024).round().astype(int)
    frame["median_ttft_s"] = pd.to_numeric(frame["median_ttft_s"], errors="coerce")
    frame["p95_ttft_s"] = pd.to_numeric(frame["p95_ttft_s"], errors="coerce")
    frame["median_total_latency_s"] = pd.to_numeric(
        frame["median_total_latency_s"], errors="coerce"
    )
    frame["peak_server_rss_mb"] = pd.to_numeric(frame["peak_server_rss_mb"], errors="coerce")
    baseline = frame[frame["strategy"] == "no_cache"][
        ["workload", "prompt_len_target", "median_ttft_s"]
    ].rename(columns={"median_ttft_s": "baseline_median_ttft_s"})
    frame = frame.drop(columns=["baseline_median_ttft_s", "ttft_speedup"], errors="ignore")
    frame = frame.merge(baseline, on=["workload", "prompt_len_target"], how="left")
    frame["ttft_speedup"] = frame["baseline_median_ttft_s"] / frame["median_ttft_s"]
    return frame


def plot_ttft_by_length(main: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85), sharey=True)
    for ax, workload in zip(axes, ["coding", "rag"]):
        data = main[main["workload"] == workload].sort_values("length_k")
        for strategy in ["no_cache", "default_cache", "stable_prefix"]:
            series = data[data["strategy"] == strategy]
            ax.plot(
                series["length_k"],
                series["median_ttft_s"],
                color=STRATEGY_COLORS[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=7.4,
                linewidth=2.8,
                label=STRATEGY_LABELS[strategy],
            )
            for _, row in series.iterrows():
                if row["length_k"] in (2, 16) and strategy != "default_cache":
                    ax.annotate(
                        seconds_formatter(row["median_ttft_s"], 0),
                        (row["length_k"], row["median_ttft_s"]),
                        xytext=(0, 9 if strategy == "stable_prefix" else -14),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9.5,
                        fontweight="bold",
                        color=STRATEGY_COLORS[strategy],
                    )
        ax.set_title(WORKLOAD_LABELS[workload], loc="left", pad=10)
        ax.set_xlabel("Prompt length target")
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2K", "4K", "8K", "16K"])
        ax.set_yscale("log")
        ax.set_ylim(0.075, 60)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        polish_axis(ax)
    axes[0].set_ylabel("Warm TTFT (s)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.01), ncol=3)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.83, bottom=0.19, wspace=0.12)
    save_figure(fig, output_dir, "report_ttft_by_length")


def plot_speedup_by_length(main: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), sharey=True)
    for ax, workload in zip(axes, ["coding", "rag"]):
        data = main[
            (main["workload"] == workload) & (main["strategy"].isin(["default_cache", "stable_prefix"]))
        ].sort_values("length_k")
        for strategy in ["default_cache", "stable_prefix"]:
            series = data[data["strategy"] == strategy]
            ax.plot(
                series["length_k"],
                series["ttft_speedup"],
                color=STRATEGY_COLORS[strategy],
                marker=STRATEGY_MARKERS[strategy],
                markersize=7.4,
                linewidth=2.8,
                label=STRATEGY_LABELS[strategy],
            )
            if strategy == "stable_prefix":
                for _, row in series.iterrows():
                    if row["length_k"] in (2, 8, 16):
                        ax.annotate(
                            f"{row['ttft_speedup']:.0f}x",
                            (row["length_k"], row["ttft_speedup"]),
                            xytext=(0, 9),
                            textcoords="offset points",
                            ha="center",
                            fontsize=9.5,
                            fontweight="bold",
                            color=STRATEGY_COLORS[strategy],
                        )
        ax.axhline(1.0, color=SPINE, linewidth=1.2)
        ax.set_title(WORKLOAD_LABELS[workload], loc="left", pad=10)
        ax.set_xlabel("Prompt length target")
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2K", "4K", "8K", "16K"])
        ax.set_yscale("log")
        ax.set_ylim(0.7, 160)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(speedup_formatter))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        polish_axis(ax)
    axes[0].set_ylabel("Warm TTFT speedup")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 1.01), ncol=2)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.83, bottom=0.19, wspace=0.12)
    save_figure(fig, output_dir, "report_speedup_by_length")


def plot_layout_sensitivity(summary_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(summary_dir / "formal_layout_coding_rag_8k_summary.csv")
    frame["median_ttft_s"] = pd.to_numeric(frame["median_ttft_s"], errors="coerce")
    order = ["front_volatile", "middle_volatile", "end_volatile", "stable_prefix"]
    y_positions = {layout: len(order) - idx for idx, layout in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), sharex=True)
    for ax, workload in zip(axes, ["coding", "rag"]):
        data = frame[frame["workload"] == workload].set_index("layout").loc[order].reset_index()
        ax.hlines(
            [y_positions[item] for item in order],
            xmin=0,
            xmax=data["median_ttft_s"],
            color="#D8DEE8",
            linewidth=5,
            zorder=1,
        )
        for _, row in data.iterrows():
            y = y_positions[row["layout"]]
            ax.scatter(
                row["median_ttft_s"],
                y,
                s=115,
                color=LAYOUT_COLORS[row["layout"]],
                edgecolor="white",
                linewidth=1.4,
                zorder=3,
            )
            offset = 0.26 if row["median_ttft_s"] >= 1 else 0.32
            ax.text(
                row["median_ttft_s"] + offset,
                y,
                seconds_formatter(row["median_ttft_s"], 0),
                va="center",
                fontsize=9.8,
                fontweight="bold",
                color=TEXT,
            )
        ax.set_title(WORKLOAD_LABELS[workload], loc="left", pad=10)
        ax.set_yticks([y_positions[item] for item in order])
        ax.set_yticklabels([LAYOUT_LABELS[item] for item in order])
        ax.set_xlabel("Warm TTFT (s)")
        ax.set_xlim(0, 13.8)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        polish_axis(ax)
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.19, wspace=0.24)
    save_figure(fig, output_dir, "report_layout_sensitivity_8k")


def plot_cache_reuse(summary_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(summary_dir / "formal_cache_reuse_coding_rag_8k_16k_summary.csv")
    frame["cache_reuse"] = pd.to_numeric(frame["cache_reuse"], errors="coerce")
    frame["median_ttft_s"] = pd.to_numeric(frame["median_ttft_s"], errors="coerce")
    frame["speedup_vs_reuse0"] = pd.to_numeric(
        frame["ttft_speedup_vs_reuse0"], errors="coerce"
    )
    frame["length_k"] = (pd.to_numeric(frame["prompt_len_target"], errors="coerce") / 1024).round().astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharex=True, sharey=True)

    colors = {"coding": "#2563EB", "rag": "#008E7D"}
    panels = [("coding", 8), ("coding", 16), ("rag", 8), ("rag", 16)]
    for ax, (workload, length_k) in zip(axes, panels):
        ax.set_title(
            f"{WORKLOAD_LABELS[workload].replace(' agent', '')} {length_k}K",
            loc="left",
            pad=6,
            fontsize=8.5,
        )
        data = frame[(frame["workload"] == workload) & (frame["length_k"] == length_k)].sort_values(
            "cache_reuse"
        )
        ax.axhspan(0.95, 1.05, color="#EEF2F7", zorder=0)
        ax.axhline(1.0, color=SPINE, linewidth=1.2, linestyle="--")
        ax.plot(
            data["cache_reuse"],
            data["speedup_vs_reuse0"],
            color=colors[workload],
            marker="o",
            markersize=5.8,
            linewidth=2.2,
        )
        for _, point in data.iterrows():
            if point["cache_reuse"] in (0, 512):
                ax.annotate(
                    f"{point['speedup_vs_reuse0']:.2f}x",
                    (point["cache_reuse"], point["speedup_vs_reuse0"]),
                    xytext=(0, 8 if point["speedup_vs_reuse0"] >= 1 else -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.8,
                    fontweight="bold",
                    color=colors[workload],
                )
        ax.set_ylim(0.84, 1.08)
        ax.set_xticks([0, 128, 256, 512])
        ax.tick_params(axis="x", labelsize=8.2)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        polish_axis(ax)
    fig.supxlabel("cache-reuse value", y=0.075, fontsize=8.5, fontweight="bold", color=TEXT)
    fig.supylabel("Speedup vs reuse=0", x=0.01, fontsize=8.5, fontweight="bold", color=TEXT)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.86, bottom=0.23, wspace=0.16)
    save_figure(fig, output_dir, "report_cache_reuse_sweep")


def plot_memory(main: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), sharey=True)
    for ax, workload in zip(axes, ["coding", "rag"]):
        data = main[main["workload"] == workload].copy()
        data["rss_gb"] = data["peak_server_rss_mb"] / 1024
        offsets = {"no_cache": -0.32, "default_cache": 0.0, "stable_prefix": 0.32}
        for strategy in ["no_cache", "default_cache", "stable_prefix"]:
            series = data[data["strategy"] == strategy].sort_values("length_k")
            ax.bar(
                series["length_k"] + offsets[strategy],
                series["rss_gb"],
                width=0.29,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
                alpha=0.92,
            )
        ax.set_title(WORKLOAD_LABELS[workload], loc="left", pad=6, fontsize=8.5)
        ax.set_xlabel("Prompt length target", fontsize=8.5)
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2K", "4K", "8K", "16K"])
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        polish_axis(ax)
    axes[0].set_ylabel("Peak RSS (GiB)", fontsize=8.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.995), ncol=3, fontsize=8.5)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.83, bottom=0.19, wspace=0.12)
    save_figure(fig, output_dir, "report_memory_by_strategy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", default="results/summary")
    parser.add_argument("--output-dir", default="results/figures/report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)
    main_summary = read_main_summary(summary_dir)
    plot_ttft_by_length(main_summary, output_dir)
    plot_speedup_by_length(main_summary, output_dir)
    plot_layout_sensitivity(summary_dir, output_dir)
    plot_cache_reuse(summary_dir, output_dir)
    plot_memory(main_summary, output_dir)
    for path in sorted(output_dir.glob("report_*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
