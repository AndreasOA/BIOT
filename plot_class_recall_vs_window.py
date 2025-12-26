#!/usr/bin/env python3
"""plot_class_recall_vs_window.py

Create a thesis-friendly figure showing per-class recall vs. window length,
for multiple architectures, using the aggregated confusion-matrix LaTeX tables
in ./confusion_matrix_analysis.

Figure concept (requested):
- One subplot per class (0–5)
- X-axis: window length (5s, 7s, 9s)
- Y-axis: recall (0–1), shared across subplots
- One line per architecture
- Shading: ± 1 SD across seeds

Data source:
- confusion_matrix_analysis/cm_table_<ARCH>_±2s_±2s.tex  -> 5s
- confusion_matrix_analysis/cm_table_<ARCH>_±3s_±3s.tex  -> 7s
- confusion_matrix_analysis/cm_table_<ARCH>_±4s_±4s.tex  -> 9s

Usage:
  python plot_class_recall_vs_window.py --no-show

Optional (experimental):
  --annotate-significance will add '*' markers where a per-class recall
  difference vs. Linear Transformer is significant (min(p_t, p_w) < alpha)
  based on comparison_class_Linear_Transformer_..._vs_<ARCH>_....tex files.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RecallPoint:
    mean: float
    std: float


_ARCH_LABELS = {
    "Linear_Transformer": "Linear Transformer",
    "mLSTM": "mLSTM",
    "sLSTM": "sLSTM",
    "mLSTM_sLSTM": "mLSTM+sLSTM",
}


# Class id -> (acronym, full name)
# In this repo, TUEV labels are stored as 1..6 in the .rec/.pkl files and converted to 0..5 via (label - 1).
_CLASS_INFO: Dict[int, Tuple[str, str]] = {
    0: ("SPSW", "Spike and Sharp Wave"),
    1: ("GPED", "Generalized Periodic Epileptiform Discharge"),
    2: ("PLED", "Periodic Lateralized Epileptiform Discharge"),
    3: ("EYEM", "Eye Movement"),
    4: ("ARTF", "Artifact"),
    5: ("BCKG", "Background"),
}


def _class_title(cls: int) -> str:
    abbr, full = _CLASS_INFO.get(cls, (f"C{cls}", f"Class {cls}"))
    return f"Class {cls}: {full} ({abbr})"


def _compute_ylim(values: Iterable[float], pad: float = 0.05) -> Tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    if hi - lo < 0.10:
        mid = (hi + lo) / 2.0
        lo = max(0.0, mid - 0.05)
        hi = min(1.0, mid + 0.05)
    return lo, hi


def _sec_to_window_s(half_window_s: int) -> int:
    # File naming uses ±Ns; in this project that maps to a (2N+1)s window.
    return 2 * half_window_s + 1


def _parse_mean_std_from_cell(cell: str) -> RecallPoint:
    """Parse a LaTeX cell like '\\textbf{0.082}$\\pm$0.164' into floats."""
    # Remove common LaTeX wrappers and math markers.
    cleaned = cell.strip()
    cleaned = cleaned.replace("\\textbf{", "").replace("}", "")
    cleaned = cleaned.replace("$\\pm$", "±")
    cleaned = cleaned.replace("\\pm", "±")
    cleaned = cleaned.replace("{\\scriptsize", "").replace("}", "")

    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", cleaned)
    if len(numbers) < 2:
        raise ValueError(f"Could not parse mean±std from cell: {cell!r}")
    return RecallPoint(mean=float(numbers[0]), std=float(numbers[1]))


_CM_FILENAME_RE = re.compile(
    r"^cm_table_(?P<arch>.+?)_±(?P<sec>\d+)s_±(?P=sec)s\.tex$"
)


def _architecture_label_from_token(token: str) -> str:
    if token in _ARCH_LABELS:
        return _ARCH_LABELS[token]
    # Fallback: replace underscores for any unseen arch names.
    return token.replace("_", " ")


def parse_confusion_matrix_table(tex_path: str) -> Dict[int, RecallPoint]:
    """Return per-class diagonal (recall) mean±std from a cm_table_*.tex."""
    if not os.path.exists(tex_path):
        raise FileNotFoundError(tex_path)

    per_class: Dict[int, RecallPoint] = {}

    # Row example:
    # \textbf{Class 3} & ... & \textbf{0.664}$\pm$0.046 & ... \\
    row_re = re.compile(r"^\\textbf\{Class\s+(?P<cls>\d+)\}\s*&\s*(?P<rest>.+?)\\\\\s*$")

    with open(tex_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            m = row_re.match(line)
            if not m:
                continue

            cls = int(m.group("cls"))
            # Split remaining columns by &
            cols = [c.strip() for c in m.group("rest").split("&")]
            # The diagonal is column index == cls (0-based) because cols excludes the row label.
            if cls < 0 or cls >= len(cols):
                raise ValueError(f"Unexpected column count in {tex_path}: class={cls}, cols={len(cols)}")

            per_class[cls] = _parse_mean_std_from_cell(cols[cls])

    if len(per_class) != 6:
        raise ValueError(f"Expected 6 classes in {tex_path}, parsed {len(per_class)}")

    return per_class


def load_recall_grid(cm_dir: str) -> Tuple[list[str], list[int], Dict[Tuple[str, int, int], RecallPoint]]:
    """Load recall(mean,std) keyed by (arch_label, window_s, class_id)."""
    if not os.path.isdir(cm_dir):
        raise NotADirectoryError(cm_dir)

    data: Dict[Tuple[str, int, int], RecallPoint] = {}
    archs_found: set[str] = set()
    windows_found: set[int] = set()

    for name in os.listdir(cm_dir):
        m = _CM_FILENAME_RE.match(name)
        if not m:
            continue

        arch_token = m.group("arch")
        half_window = int(m.group("sec"))
        window_s = _sec_to_window_s(half_window)
        arch_label = _architecture_label_from_token(arch_token)

        per_class = parse_confusion_matrix_table(os.path.join(cm_dir, name))
        for cls, rp in per_class.items():
            data[(arch_label, window_s, cls)] = rp

        archs_found.add(arch_label)
        windows_found.add(window_s)

    architectures = [
        "Linear Transformer",
        "mLSTM",
        "sLSTM",
        "mLSTM+sLSTM",
    ]
    # Keep only the ones present.
    architectures = [a for a in architectures if a in archs_found]

    windows = sorted(windows_found)

    if not data:
        raise ValueError(f"No cm_table_*.tex files found in {cm_dir}")

    return architectures, windows, data


def _get_architecture_name_from_flags(mlstm: bool, slstm: bool) -> str:
    if not mlstm and not slstm:
        return "Linear Transformer"
    if mlstm and not slstm:
        return "mLSTM"
    if not mlstm and slstm:
        return "sLSTM"
    if mlstm and slstm:
        return "mLSTM+sLSTM"
    return "Unknown"


_RUN_FOLDER_RE = re.compile(
    r"mlstm=(?P<mlstm>True|False).*?slstm=(?P<slstm>True|False).*?secondsBeforeEvent=(?P<before>\d+).*?secondsAfterEvent=(?P<after>\d+).*?seed=(?P<seed>\d+)"
)


def _load_confusion_matrix_csv(csv_path: Path) -> np.ndarray:
    # confusion_matrix.csv has a header column/row.
    return np.genfromtxt(csv_path, delimiter=",", skip_header=1)[:, 1:].astype(float)


def _normalize_confusion_matrix_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return cm / row_sums


def load_recall_runs_grid(
    runs_root: str,
) -> Tuple[list[str], list[int], Dict[Tuple[str, int, int], list[float]]]:
    """Load per-seed per-class recall lists.

    Returns:
        architectures, windows, runs
    Where runs is keyed by (arch_label, window_s, class_id) -> list of recall values.
    """

    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(str(root))

    runs: Dict[Tuple[str, int, int], list[float]] = {}
    archs_found: set[str] = set()
    windows_found: set[int] = set()

    # Expected structure: stored_runs/final_results/<folder>/checkpoints/bal_acc/confusion_matrix.csv
    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        m = _RUN_FOLDER_RE.search(folder.name)
        if not m:
            continue

        mlstm = m.group("mlstm") == "True"
        slstm = m.group("slstm") == "True"
        before = int(m.group("before"))
        after = int(m.group("after"))
        window_s = before + after + 1

        arch = _get_architecture_name_from_flags(mlstm, slstm)
        if arch == "Unknown":
            continue

        csv_path = folder / "checkpoints" / "bal_acc" / "confusion_matrix.csv"
        if not csv_path.exists():
            continue

        cm = _load_confusion_matrix_csv(csv_path)
        cmn = _normalize_confusion_matrix_rows(cm)

        # Diagonal is per-class recall.
        if cmn.shape[0] < 6 or cmn.shape[1] < 6:
            continue

        for cls in range(6):
            key = (arch, window_s, cls)
            runs.setdefault(key, []).append(float(cmn[cls, cls]))

        archs_found.add(arch)
        windows_found.add(window_s)

    architectures = ["Linear Transformer", "mLSTM", "sLSTM", "mLSTM+sLSTM"]
    architectures = [a for a in architectures if a in archs_found]
    windows = sorted(windows_found)

    return architectures, windows, runs


_COMPARISON_RE = re.compile(
    r"^comparison_class_Linear_Transformer_±(?P<sec>\d+)s_±(?P=sec)s_vs_(?P<arch>.+?)_±(?P=sec)s_±(?P=sec)s\.tex$"
)


def load_significance_vs_linear(
    cm_dir: str,
    alpha: float,
) -> Dict[Tuple[str, int, int], bool]:
    """Parse comparison_class_Linear_Transformer_*_vs_<ARCH>_* tables.

    Returns dict keyed by (arch_label, window_s, class_id) -> significant.
    Significant if min(p_t, p_w) < alpha.
    """

    sig: Dict[Tuple[str, int, int], bool] = {}

    row_re = re.compile(
        r"^Class\s+(?P<cls>\d+)\s*&\s*(?P<a>[^&]+?)\s*&\s*(?P<b>[^&]+?)\s*&\s*(?P<tstat>[-+]?\d+(?:\.\d+)?)\s*&\s*(?P<pt>\d+(?:\.\d+)?)\s*&\s*(?P<pw>\d+(?:\.\d+)?)\s*\\\\\s*$"
    )

    for name in os.listdir(cm_dir):
        m = _COMPARISON_RE.match(name)
        if not m:
            continue

        half_window = int(m.group("sec"))
        window_s = _sec_to_window_s(half_window)
        arch_label = _architecture_label_from_token(m.group("arch"))

        path = os.path.join(cm_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                mm = row_re.match(line)
                if not mm:
                    continue

                cls = int(mm.group("cls"))
                pt = float(mm.group("pt"))
                pw = float(mm.group("pw"))
                sig[(arch_label, window_s, cls)] = min(pt, pw) < alpha

    return sig


def plot_class_recall_vs_window(
    architectures: list[str],
    windows: list[int],
    data: Dict[Tuple[str, int, int], RecallPoint],
    colors: Dict[str, str],
    layout: str = "column",
    annotate_significance: bool = False,
    significance: Dict[Tuple[str, int, int], bool] | None = None,
    alpha: float = 0.05,
):
    x = np.array(windows, dtype=float)

    if layout == "grid":
        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(9.0, 8.6),
            sharex=True,
            sharey=False,
        )
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(
            nrows=6,
            ncols=1,
            figsize=(8.6, 14.6),
            sharex=True,
            sharey=False,
        )
        axes = axes.flatten()

    for cls in range(6):
        ax = axes[cls]

        ylim_values: list[float] = []
        for arch in architectures:
            means = np.array(
                [data.get((arch, w, cls), RecallPoint(np.nan, np.nan)).mean for w in windows],
                dtype=float,
            )
            stds = np.array(
                [data.get((arch, w, cls), RecallPoint(np.nan, np.nan)).std for w in windows],
                dtype=float,
            )

            if not np.isfinite(means).any():
                continue

            for mu, sd in zip(means, stds):
                if np.isfinite(mu) and np.isfinite(sd):
                    ylim_values.extend([mu - sd, mu + sd])

            ax.plot(
                x,
                means,
                marker="o",
                markersize=4,
                linewidth=2.0,
                color=colors[arch],
                label=arch,
            )
            ax.fill_between(
                x,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=colors[arch],
                alpha=0.15,
                linewidth=0,
            )

            if annotate_significance and significance is not None and arch != "Linear Transformer":
                for xi, w, mu, sd in zip(x, windows, means, stds):
                    if not np.isfinite(mu) or not np.isfinite(sd):
                        continue
                    if significance.get((arch, w, cls), False):
                        y_star = min(0.98, mu + sd + 0.03)
                        ax.text(
                            xi,
                            y_star,
                            "*",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color="black",
                        )

        ax.set_title(_class_title(cls))
        ax.set_ylabel("Recall")
        y0, y1 = _compute_ylim(ylim_values, pad=0.05)
        ax.set_ylim(y0, y1)
        ax.grid(True, axis="y", which="major", alpha=0.35)
        ax.grid(True, axis="y", which="minor", alpha=0.18)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Show x-axis on every subplot (shared-x otherwise hides upper tick labels).
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}s" for w in windows])
        ax.set_xlabel("Temporal Window")
        ax.tick_params(axis="x", labelbottom=True)

    # Legend as colored lines (matching plot_runtime_comparison.py)
    handles = [plt.Line2D([0], [0], color=colors[a], lw=7, alpha=0.75) for a in architectures]
    fig.legend(
        handles,
        architectures,
        loc="upper center",
        ncol=len(architectures),
        framealpha=0.9,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.988),
    )

    fig.suptitle(
        "Class Recall as a Function of Temporal Window and Architecture",
        fontsize=13,
        weight="bold",
    )

    if annotate_significance:
        fig.text(
            0.5,
            0.02,
            f"'*' indicates significant vs. Linear Transformer (min(p_t, p_w) < {alpha})",
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    return fig


def _plot_grouped_boxplots_per_class(
    architectures: list[str],
    windows: list[int],
    runs: Dict[Tuple[str, int, int], list[float]],
    colors: Dict[str, str],
    layout: str = "column",
    annotate_significance: bool = False,
    significance: Dict[Tuple[str, int, int], bool] | None = None,
    alpha: float = 0.05,
):
    x = np.arange(len(windows), dtype=float)

    if layout == "grid":
        fig, axes = plt.subplots(
            nrows=3,
            ncols=2,
            figsize=(9.0, 8.6),
            sharex=True,
            sharey=False,
        )
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(
            nrows=6,
            ncols=1,
            figsize=(8.6, 14.6),
            sharex=True,
            sharey=False,
        )
        axes = axes.flatten()

    n_arch = len(architectures)
    box_width = 0.18
    offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * (box_width + 0.02)

    for cls in range(6):
        ax = axes[cls]
        ylim_values: list[float] = []
        for i, arch in enumerate(architectures):
            data_for_arch = [runs.get((arch, w, cls), []) for w in windows]
            for vv in data_for_arch:
                ylim_values.extend(list(vv))
            positions = x + offsets[i]

            bp = ax.boxplot(
                data_for_arch,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 1.0},
                capprops={"linewidth": 1.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[arch])
                patch.set_alpha(0.55)
                patch.set_edgecolor(colors[arch])
                patch.set_linewidth(1.2)
            for element in ("whiskers", "caps"):
                for line in bp[element]:
                    line.set_color(colors[arch])
                    line.set_alpha(0.9)

            if annotate_significance and significance is not None and arch != "Linear Transformer":
                for j, w in enumerate(windows):
                    if not significance.get((arch, w, cls), False):
                        continue
                    vals = np.asarray(runs.get((arch, w, cls), []), dtype=float)
                    if vals.size == 0:
                        continue
                    y_star = float(np.nanpercentile(vals, 75)) + 0.05
                    y_star = min(0.98, y_star)
                    ax.text(
                        positions[j],
                        y_star,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="black",
                    )

        ax.set_title(_class_title(cls))
        ax.set_ylabel("Recall")
        y0, y1 = _compute_ylim(ylim_values, pad=0.05)
        ax.set_ylim(y0, y1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(True, axis="y", which="minor", alpha=0.15)

    # Show x-axis on every subplot (shared-x otherwise hides upper tick labels).
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}s" for w in windows])
        ax.set_xlabel("Temporal Window")
        ax.tick_params(axis="x", labelbottom=True)

    handles = [plt.Line2D([0], [0], color=colors[a], lw=7, alpha=0.75) for a in architectures]
    fig.legend(
        handles,
        architectures,
        loc="upper center",
        ncol=len(architectures),
        framealpha=0.9,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.97),
    )

    fig.suptitle(
        "Class Recall as a Function of Temporal Window and Architecture",
        fontsize=13,
        weight="bold",
    )

    if annotate_significance:
        fig.text(
            0.5,
            0.02,
            f"'*' indicates significant vs. Linear Transformer (min(p_t, p_w) < {alpha})",
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cm-dir",
        default="confusion_matrix_analysis",
        help="Directory containing cm_table_*.tex and comparison_class_*.tex",
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
        "--plot",
        choices=["box", "line"],
        default="box",
        help="Plot type (default: box). 'box' uses per-seed confusion_matrix.csv if available; 'line' uses cm_table_*.tex.",
    )
    parser.add_argument(
        "--layout",
        choices=["column", "grid"],
        default="column",
        help="Layout for the 6 class subplots (default: column). 'column' = one subplot per row; 'grid' = 3x2.",
    )
    parser.add_argument(
        "--runs-dir",
        default=os.path.join("stored_runs", "final_results"),
        help="Root directory that contains per-run folders with checkpoints/bal_acc/confusion_matrix.csv (default: stored_runs/final_results)",
    )
    parser.add_argument(
        "--annotate-significance",
        action="store_true",
        help="Mark per-class points that are significant vs. Linear Transformer (if comparison files exist).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold when using --annotate-significance (default: 0.05)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows.",
    )
    args = parser.parse_args()

    # Keep naming/colors consistent with plot_runtime_comparison.py
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Linear Transformer": "#4C72B0",
        "mLSTM": "#55A868",
        "sLSTM": "#C44E52",
        "mLSTM+sLSTM": "#8172B2",
    }

    significance = None
    if args.annotate_significance:
        significance = load_significance_vs_linear(args.cm_dir, alpha=args.alpha)

    fig = None
    if args.plot == "box":
        try:
            architectures, windows, runs = load_recall_runs_grid(args.runs_dir)
            if not runs or not architectures or not windows:
                raise ValueError("No per-run confusion_matrix.csv files found")
            fig = _plot_grouped_boxplots_per_class(
                architectures=architectures,
                windows=windows,
                runs=runs,
                colors=colors,
                layout=args.layout,
                annotate_significance=args.annotate_significance,
                significance=significance,
                alpha=args.alpha,
            )
        except Exception as e:
            print(f"[WARN] Boxplot mode unavailable ({e}); falling back to line plot from LaTeX tables.")
            args.plot = "line"

    if fig is None:
        architectures, windows, data = load_recall_grid(args.cm_dir)
        fig = plot_class_recall_vs_window(
            architectures=architectures,
            windows=windows,
            data=data,
            colors=colors,
            layout=args.layout,
            annotate_significance=args.annotate_significance,
            significance=significance,
            alpha=args.alpha,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    filename_base = f"class_recall_vs_window_{args.plot}"
    out_png = os.path.join(args.output_dir, f"{filename_base}.png")
    out_pdf = os.path.join(args.output_dir, f"{filename_base}.pdf")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        out_png2 = os.path.join(args.plot_dir, f"{filename_base}.png")
        out_pdf2 = os.path.join(args.plot_dir, f"{filename_base}.pdf")
        fig.savefig(out_png2, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf2, bbox_inches="tight")
        print(f"Saved: {out_png2}")
        print(f"Saved: {out_pdf2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
