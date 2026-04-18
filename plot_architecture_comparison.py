#!/usr/bin/env python3
"""plot_architecture_comparison.py

Plots performance across different architectures and temporal windows.

Default behavior (thesis-friendly):
- Plot a single main metric (default: AUCPR macro) to keep the figure readable.
- Use grouped boxplots across the 3 temporal windows (5s/7s/9s), based on the
    per-run values stored in the *_analysis.txt files.

Optional:
- Create an appendix-style multi-metric summary (line plot with std shading).
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re



def parse_metrics_analysis_file(filepath):
    """Parse a metrics *_analysis.txt file.

    Returns:
        dict: {metric_name: {'mean': float|nan, 'std': float|nan, 'runs': list[float]}}
    """
    metrics: dict[str, dict] = {}

    with open(filepath, "r") as f:
        lines = f.read().split("\n")

    # 1) OVERALL STATISTICS section
    in_stats_section = False
    for line in lines:
        if "OVERALL STATISTICS" in line:
            in_stats_section = True
            continue
        if in_stats_section and line.startswith("="):
            break
        if in_stats_section and line.strip():
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ["Metric", "------"]:
                metric_name = parts[0]
                try:
                    mean_val = float(parts[1])
                    std_val = float(parts[2])
                except (ValueError, IndexError):
                    continue
                metrics.setdefault(metric_name, {"mean": np.nan, "std": np.nan, "runs": []})
                metrics[metric_name]["mean"] = mean_val
                metrics[metric_name]["std"] = std_val

    # 2) INDIVIDUAL RUN VALUES section
    in_runs_section = False
    current_metric = None
    for line in lines:
        if "INDIVIDUAL RUN VALUES" in line:
            in_runs_section = True
            continue
        if not in_runs_section:
            continue

        if not line.strip():
            continue

        # Metric header line, e.g. "aucpr_macro:"
        if not line.startswith(" ") and line.endswith(":"):
            current_metric = line[:-1].strip()
            metrics.setdefault(current_metric, {"mean": np.nan, "std": np.nan, "runs": []})
            continue

        # Run value line, e.g. "  <run_name>: 0.4951"
        if current_metric is not None and line.startswith("  ") and ":" in line:
            try:
                value_str = line.split(":", 1)[1].strip()
                value = float(value_str)
                metrics[current_metric]["runs"].append(value)
            except ValueError:
                continue

    return metrics


def get_architecture_name(mlstm, slstm):
    """Convert mlstm/slstm flags to architecture name."""
    if mlstm == 'False' and slstm == 'False':
        return 'Linear Transformer'
    elif mlstm == 'True' and slstm == 'False':
        return 'mLSTM'
    elif mlstm == 'False' and slstm == 'True':
        return 'sLSTM'
    elif mlstm == 'True' and slstm == 'True':
        return 'mLSTM+sLSTM'
    else:
        return 'Unknown'


def load_all_metrics(metrics_dir):
    """
    Load all metrics from the metrics directory.
    
    Returns:
        dict: {architecture: {window_size: {metric: {'mean': float, 'std': float}}}}
    """
    data = {}
    
    # Pattern: metrics_BIOT_mlstm=<bool>_slstm=<bool>_secondsBeforeEvent=<n>_secondsAfterEvent=<n>_bal_acc_analysis.txt
    pattern = r'metrics_BIOT_mlstm=(\w+)_slstm=(\w+)_secondsBeforeEvent=(\d+)_secondsAfterEvent=(\d+)_bal_acc_analysis\.txt'
    
    for filename in os.listdir(metrics_dir):
        match = re.match(pattern, filename)
        if match:
            mlstm, slstm, before, after = match.groups()
            
            # Calculate total window size
            window = int(before) + int(after) + 1  # +1 for the event itself
            
            arch = get_architecture_name(mlstm, slstm)
            
            filepath = os.path.join(metrics_dir, filename)
            metrics = parse_metrics_analysis_file(filepath)
            
            if arch not in data:
                data[arch] = {}
            data[arch][window] = metrics
    
    return data


def extract_metric_arrays(data, metric_name, architectures, windows):
    """
    Extract mean and std arrays for a specific metric across architectures and windows.
    
    Returns:
        (mean_array, std_array): Both are numpy arrays of shape (n_architectures, n_windows)
    """
    means = []
    stds = []
    
    for arch in architectures:
        arch_means = []
        arch_stds = []
        for window in windows:
            if arch in data and window in data[arch] and metric_name in data[arch][window]:
                arch_means.append(data[arch][window][metric_name]['mean'])
                arch_stds.append(data[arch][window][metric_name]['std'])
            else:
                # Missing data - use NaN
                arch_means.append(np.nan)
                arch_stds.append(np.nan)
        means.append(arch_means)
        stds.append(arch_stds)
    
    return np.array(means), np.array(stds)


def extract_metric_runs(data, metric_name, architectures, windows):
    """Extract per-run lists for a metric across architectures and windows.

    Returns:
        list[list[list[float]]]: shape (n_architectures, n_windows)
    """
    runs: list[list[list[float]]] = []
    for arch in architectures:
        arch_runs: list[list[float]] = []
        for window in windows:
            if (
                arch in data
                and window in data[arch]
                and metric_name in data[arch][window]
                and data[arch][window][metric_name].get("runs")
            ):
                arch_runs.append(list(data[arch][window][metric_name]["runs"]))
            else:
                arch_runs.append([])
        runs.append(arch_runs)
    return runs


def plot_grouped_boxplot(metric_runs, architectures, windows, colors, metric_label, ylim=None):
    """Grouped boxplots: x = window, hue = architecture."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    x = np.arange(len(windows), dtype=float)
    n_arch = len(architectures)
    box_width = 0.18
    offsets = (np.arange(n_arch) - (n_arch - 1) / 2.0) * (box_width + 0.02)

    for i, arch in enumerate(architectures):
        data_for_arch = [metric_runs[i][j] for j in range(len(windows))]
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

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}s" for w in windows])
    ax.set_xlabel("Temporal Window")
    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(ylim)

    handles = [
        plt.Line2D([0], [0], color=colors[arch], lw=6, alpha=0.7)
        for arch in architectures
    ]
    ax.legend(handles, architectures, loc="best", framealpha=0.9, fontsize=9)
    # Layout is finalized by caller (so suptitle never overlaps)
    return fig, ax


def plot_appendix_multi_metric_lines(metrics, architectures, windows, colors, shading_alpha=0.10):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()

    for ax, (metric_name, (mean, std, ylim)) in zip(axes, metrics.items()):
        for i, arch in enumerate(architectures):
            ax.plot(
                windows,
                mean[i],
                marker="o",
                markersize=6,
                lw=2,
                label=arch,
                color=colors[arch],
            )
            ax.fill_between(
                windows,
                mean[i] - std[i],
                mean[i] + std[i],
                alpha=shading_alpha,
                color=colors[arch],
            )
        ax.set_title(metric_name, fontsize=12, weight="bold")
        ax.set_xlabel("Temporal Window (s)", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_xticks(windows)
        ax.set_ylim(ylim)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Performance Summary Across Architectures and Window Lengths",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="aucpr_macro",
        help="Metric key from *_analysis.txt (default: aucpr_macro)",
    )
    parser.add_argument(
        "--plot",
        choices=["box", "line"],
        default="box",
        help="Plot type for the main figure (default: box)",
    )
    parser.add_argument(
        "--appendix",
        action="store_true",
        help="Also create the older multi-metric line summary (appendix figure).",
    )
    parser.add_argument(
        "--generate-all-metrics",
        action="store_true",
        help=(
            "Generate one figure per main metric (balanced_accuracy, cohen_kappa, "
            "f1_weighted, aucpr_macro). Intended for adding to the LaTeX docs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/workspaces/BIOT/Docs/images",
        help="Where to save generated figures (default: Docs/images).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows (recommended for batch generation).",
    )
    parser.add_argument(
        "--metrics-dir",
        default="/workspaces/BIOT/stored_runs/final_results/metrics",
        help="Directory containing metrics *_analysis.txt files.",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    
    # Load all data
    print("Loading metrics from files...")
    data = load_all_metrics(metrics_dir)
    
    # Print loaded data summary
    print("\nLoaded data:")
    for arch in sorted(data.keys()):
        windows = sorted(data[arch].keys())
        print(f"  {arch}: windows = {windows}")
    
    # Configuration
    architectures = ["Linear Transformer", "mLSTM", "sLSTM", "mLSTM+sLSTM"]
    windows = np.array([5, 7, 9])
    
    # Extract main metric
    main_metric = args.metric
    main_runs = extract_metric_runs(data, main_metric, architectures, windows)
    main_means, main_stds = extract_metric_arrays(data, main_metric, architectures, windows)
    
    # Plot aesthetics
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Linear Transformer": "#4C72B0",
        "mLSTM": "#55A868",
        "sLSTM": "#C44E52",
        "mLSTM+sLSTM": "#8172B2",
    }

    # Human-friendly metric label (fallback to raw key)
    metric_labels = {
        "balanced_accuracy": "Balanced Accuracy",
        "cohen_kappa": "Cohen's κ",
        "f1_weighted": "Weighted F1",
        "aucpr_macro": "AUCPR (Macro)",
        "aucpr_micro": "AUCPR (Micro)",
        "auroc_macro_ovr": "AUROC (Macro, OvR)",
        "auroc_weighted_ovr": "AUROC (Weighted, OvR)",
        "accuracy": "Accuracy",
    }
    metric_label = metric_labels.get(main_metric, main_metric)

    os.makedirs("/workspaces/BIOT/fig", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    def save_main_figure(fig, filename_base: str):
        # Ensure suptitle has dedicated space, and never overlaps the axes.
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_fig = f"/workspaces/BIOT/fig/{filename_base}.png"
        out_docs = os.path.join(args.output_dir, f"{filename_base}.png")
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        fig.savefig(out_docs, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_fig}")
        print(f"Saved: {out_docs}")

    def make_single_metric_figure(metric_key: str, plot_kind: str):
        runs = extract_metric_runs(data, metric_key, architectures, windows)
        means, stds = extract_metric_arrays(data, metric_key, architectures, windows)
        label = metric_labels.get(metric_key, metric_key)

        if plot_kind == "box":
            fig, _ = plot_grouped_boxplot(
                metric_runs=runs,
                architectures=architectures,
                windows=windows,
                colors=colors,
                metric_label=label,
            )
            fig.suptitle(
                f"{label} Across Architectures and Window Lengths",
                fontsize=13,
                weight="bold",
            )
            return fig

        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        for i, arch in enumerate(architectures):
            ax.plot(
                windows,
                means[i],
                marker="o",
                markersize=6,
                lw=2,
                label=arch,
                color=colors[arch],
            )
            ax.fill_between(
                windows,
                means[i] - stds[i],
                means[i] + stds[i],
                alpha=0.10,
                color=colors[arch],
            )
        ax.set_title(f"{label} (mean ± std)", fontsize=12, weight="bold")
        ax.set_xlabel("Temporal Window (s)")
        ax.set_ylabel(label)
        ax.set_xticks(windows)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        fig.suptitle(
            f"{label} Across Architectures and Window Lengths",
            fontsize=13,
            weight="bold",
        )
        return fig

    # Batch generation for docs (one file per metric)
    if args.generate_all_metrics:
        thesis_metrics = ["balanced_accuracy", "cohen_kappa", "f1_weighted", "aucpr_macro"]
        for metric_key in thesis_metrics:
            fig = make_single_metric_figure(metric_key=metric_key, plot_kind=args.plot)
            save_main_figure(fig, filename_base=f"architecture_window_{metric_key}_{args.plot}")
            plt.close(fig)

        if not args.no_show:
            print("Note: --generate-all-metrics is best used with --no-show")
        return

    # Single-figure mode
    fig = make_single_metric_figure(metric_key=main_metric, plot_kind=args.plot)
    save_main_figure(fig, filename_base=f"architecture_window_{main_metric}_{args.plot}")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # Optional appendix figure: multi-metric summary
    if args.appendix:
        bal_acc, bal_acc_std = extract_metric_arrays(data, "balanced_accuracy", architectures, windows)
        kappa, kappa_std = extract_metric_arrays(data, "cohen_kappa", architectures, windows)
        f1, f1_std = extract_metric_arrays(data, "f1_weighted", architectures, windows)
        aucpr, aucpr_std = extract_metric_arrays(data, "aucpr_macro", architectures, windows)

        appendix_metrics = {
            "Balanced Accuracy": (bal_acc, bal_acc_std, [0.35, 0.6]),
            "Cohen's κ": (kappa, kappa_std, [0.35, 0.6]),
            "Weighted F1": (f1, f1_std, [0.65, 0.8]),
            "AUCPR (Macro)": (aucpr, aucpr_std, [0.4, 0.55]),
        }
        fig = plot_appendix_multi_metric_lines(
            metrics=appendix_metrics,
            architectures=architectures,
            windows=windows,
            colors=colors,
            shading_alpha=0.10,
        )
        output_path = "/workspaces/BIOT/fig/architecture_window_all_metrics_appendix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Appendix figure saved to: {output_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
