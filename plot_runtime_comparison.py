#!/usr/bin/env python3
"""plot_runtime_comparison.py

Create a thesis-friendly runtime chart from the LaTeX runtime table.

- Input: results_runtime_table.tex (LaTeX table with mean runtime and std)
- Output: PNG saved to ./Docs/images (by default). Optionally also saved to a
    secondary directory (e.g., ./fig) if --plot-dir is provided.

The plot matches the style of plot_architecture_comparison.py:
- grouped bars across window lengths (5s/7s/9s)
- color = architecture
- error bars = std in minutes

Usage (recommended):
  python plot_runtime_comparison.py --no-show
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


@dataclass(frozen=True)
class RuntimeRow:
    architecture: str
    window_s: int
    n_runs: int
    mean_minutes: float
    std_minutes: float


_TIME_RE = re.compile(r"^\s*(?:(?P<h>\d+)\s*h)?\s*(?:(?P<m>\d+)\s*m)?\s*$")


def _parse_time_to_minutes(value: str) -> float:
    """Parse strings like '1h 49m', '5h 30m', '49m' into minutes."""
    m = _TIME_RE.match(value.strip())
    if not m:
        raise ValueError(f"Unsupported time format: {value!r}")
    hours = int(m.group("h")) if m.group("h") else 0
    minutes = int(m.group("m")) if m.group("m") else 0
    return float(hours * 60 + minutes)


def parse_runtime_table_tex(tex_path: str) -> list[RuntimeRow]:
    """Parse the LaTeX runtime table and return rows."""
    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"Runtime table not found: {tex_path}")

    rows: list[RuntimeRow] = []

    # Example row:
    # Linear Transformer & 5 & 5 & 1h 49m & 2.7 \\
    row_re = re.compile(
        r"^\s*(?P<arch>[^&]+?)\s*&\s*(?P<window>\d+)\s*&\s*(?P<n>\d+)\s*&\s*(?P<mean>[^&]+?)\s*&\s*(?P<std>[-+]?\d+(?:\.\d+)?)\s*\\\\\s*$"
    )

    with open(tex_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.startswith("\\hline"):
                continue
            m = row_re.match(line)
            if not m:
                continue

            arch = m.group("arch").strip()
            window_s = int(m.group("window"))
            n_runs = int(m.group("n"))
            mean_str = m.group("mean").strip()
            std_minutes = float(m.group("std"))

            mean_minutes = _parse_time_to_minutes(mean_str)
            rows.append(
                RuntimeRow(
                    architecture=arch,
                    window_s=window_s,
                    n_runs=n_runs,
                    mean_minutes=mean_minutes,
                    std_minutes=std_minutes,
                )
            )

    if not rows:
        raise ValueError(
            "No runtime rows parsed. Check that the input file contains LaTeX rows like: "
            "Architecture & 5 & 5 & 1h 49m & 2.7 \\\\" 
        )

    return rows


def plot_grouped_runtime_bars(
    rows: list[RuntimeRow],
    architectures: list[str],
    windows: list[int],
    colors: dict[str, str],
    y_unit: str = "hours",
):
    """Grouped bar chart: x = window, hue = architecture, y = mean runtime."""

    # Build lookup: (arch, window) -> (mean, std)
    lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for r in rows:
        lookup[(r.architecture, r.window_s)] = (r.mean_minutes, r.std_minutes)

    def convert(minutes: float) -> float:
        if y_unit == "minutes":
            return minutes
        if y_unit == "hours":
            return minutes / 60.0
        raise ValueError(f"Unsupported y_unit: {y_unit}")

    means = np.full((len(architectures), len(windows)), np.nan, dtype=float)
    stds = np.full((len(architectures), len(windows)), np.nan, dtype=float)

    for i, arch in enumerate(architectures):
        for j, w in enumerate(windows):
            if (arch, w) in lookup:
                mean_m, std_m = lookup[(arch, w)]
                means[i, j] = convert(mean_m)
                stds[i, j] = convert(std_m)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    x = np.arange(len(windows), dtype=float)
    n_arch = len(architectures)
    bar_width = 0.18
    offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * (bar_width + 0.02)

    # Make std markers clearly readable on top of semi-transparent bars.
    error_kw = {
        "elinewidth": 1.8,
        "capthick": 1.8,
        "alpha": 1.0,
        "zorder": 4,
    }

    for i, arch in enumerate(architectures):
        ax.bar(
            x + offsets[i],
            means[i],
            width=bar_width,
            color=colors[arch],
            alpha=0.70,
            edgecolor=colors[arch],
            linewidth=1.0,
            label=arch,
            yerr=stds[i],
            capsize=5,
            ecolor="black",
            error_kw=error_kw,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}s" for w in windows])
    ax.set_xlabel("Temporal Window")
    ax.set_ylabel("Runtime (hours)" if y_unit == "hours" else "Runtime (min)")

    # Finer y-axis scale to better read bar heights.
    finite_vals = means[np.isfinite(means)]
    if finite_vals.size:
        max_y = float(np.nanmax(finite_vals))
        ax.set_ylim(0.0, max_y * 1.12)

    if y_unit == "hours":
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(30))
        ax.yaxis.set_minor_locator(MultipleLocator(15))

    ax.grid(True, axis="y", which="major", alpha=0.35)
    ax.grid(True, axis="y", which="minor", alpha=0.18)

    # Legend as colored lines (matches other script's look)
    handles = [plt.Line2D([0], [0], color=colors[a], lw=6, alpha=0.7) for a in architectures]
    ax.legend(handles, architectures, loc="best", framealpha=0.9, fontsize=9)

    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-tex",
        default="results_runtime_table.tex",
        help="Path to LaTeX runtime table (default: results_runtime_table.tex)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Docs", "images"),
        help="Where to save generated figures (default: Docs/images)",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional secondary output directory (e.g., fig). If omitted, only --output-dir is used.",
    )
    parser.add_argument(
        "--y-unit",
        choices=["hours", "minutes"],
        default="hours",
        help="Y-axis unit (default: hours)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows.",
    )
    args = parser.parse_args()

    rows = parse_runtime_table_tex(args.input_tex)

    # Keep naming consistent with plot_architecture_comparison.py
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Linear Transformer": "#4C72B0",
        "mLSTM": "#55A868",
        "sLSTM": "#C44E52",
        "mLSTM+sLSTM": "#8172B2",
    }

    # Architecture naming in results_runtime_table.tex is without \ac{}
    architectures = ["Linear Transformer", "mLSTM", "sLSTM", "mLSTM+sLSTM"]
    windows = [5, 7, 9]

    fig, _ = plot_grouped_runtime_bars(
        rows=rows,
        architectures=architectures,
        windows=windows,
        colors=colors,
        y_unit=args.y_unit,
    )
    fig.suptitle(
        "Training Runtime Across Architectures and Window Lengths",
        fontsize=13,
        weight="bold",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    filename_base = f"architecture_window_runtime_bar_{args.y_unit}"
    out_docs = os.path.join(args.output_dir, f"{filename_base}.png")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_docs, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_docs}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        out_fig = os.path.join(args.plot_dir, f"{filename_base}.png")
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_fig}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
