#!/usr/bin/env python3
"""
Script to extract metrics from analysis files and create plots showing
how performance changes across different temporal window lengths (5s, 7s, 9s).

The script creates publication-ready plots suitable for LaTeX documents.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set up matplotlib for LaTeX compatibility
plt.rcParams.update({
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

def parse_filename(filename):
    """
    Extract configuration parameters from filename.
    
    Example filename: metrics_BIOT_mlstm=False_slstm=True_secondsBeforeEvent=3_secondsAfterEvent=3_bal_acc_analysis.txt
    """
    pattern = r'metrics_BIOT_mlstm=(\w+)_slstm=(\w+)_secondsBeforeEvent=(\d+)_secondsAfterEvent=(\d+)_bal_acc_analysis\.txt'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    mlstm = match.group(1) == 'True'
    slstm = match.group(2) == 'True'
    seconds_before = int(match.group(3))
    seconds_after = int(match.group(4))
    
    # Determine architecture type
    if not mlstm and not slstm:
        architecture = 'Linear Transformer'
    elif mlstm and not slstm:
        architecture = 'mLSTM'
    elif not mlstm and slstm:
        architecture = 'sLSTM'
    else:  # both True
        architecture = 'mLSTM + sLSTM'
    
    # Calculate total window length
    window_length = seconds_before + seconds_after
    
    return {
        'architecture': architecture,
        'window_length': window_length,
        'mlstm': mlstm,
        'slstm': slstm,
        'seconds_before': seconds_before,
        'seconds_after': seconds_after
    }

def extract_metrics_from_file(filepath):
    """
    Extract metric statistics from an analysis file.
    """
    metrics = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the metrics section
    in_metrics_section = False
    for line in lines:
        line = line.strip()
        
        # Start of metrics section
        if 'OVERALL STATISTICS ACROSS 5 RUNS' in line:
            in_metrics_section = True
            continue
        
        # End of metrics section
        if line.startswith('======') and in_metrics_section:
            break
        
        # Parse metric lines
        if in_metrics_section and line and not line.startswith('-') and not line.startswith('Metric'):
            parts = line.split()
            if len(parts) >= 4:
                metric_name = parts[0]
                mean_val = float(parts[1])
                std_val = float(parts[2])
                cv_val = float(parts[3])
                
                metrics[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val
                }
    
    return metrics

def load_all_metrics(results_dir):
    """
    Load metrics from all analysis files in the results directory.
    """
    all_data = []
    
    results_path = Path(results_dir)
    
    for filename in os.listdir(results_path):
        if filename.endswith('_analysis.txt'):
            # Parse filename to get configuration
            config = parse_filename(filename)
            if config is None:
                print(f"Warning: Could not parse filename {filename}")
                continue
            
            # Extract metrics from file
            filepath = results_path / filename
            metrics = extract_metrics_from_file(filepath)
            
            # Combine configuration and metrics
            for metric_name, metric_data in metrics.items():
                row = {
                    'architecture': config['architecture'],
                    'window_length': config['window_length'],
                    'metric_name': metric_name,
                    'mean': metric_data['mean'],
                    'std': metric_data['std'],
                    'cv': metric_data['cv']
                }
                all_data.append(row)
    
    return pd.DataFrame(all_data)

def create_metric_plots(df, output_dir='fig'):
    """
    Create plots for key metrics across window lengths and architectures.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics to plot
    key_metrics = ['balanced_accuracy', 'cohen_kappa', 'f1_weighted', 'auroc_macro_ovr', 'aucpr_macro']
    
    # Metric display names for plots
    metric_labels = {
        'balanced_accuracy': 'Balanced Accuracy',
        'cohen_kappa': "Cohen's κ",
        'f1_weighted': 'Weighted F1-Score',
        'auroc_macro_ovr': 'AUROC (Macro)',
        'aucpr_macro': 'AUCPR (Macro)'
    }
    
    # Architecture colors and styles
    arch_colors = {
        'Linear Transformer': '#1f77b4',  # Blue
        'mLSTM': '#ff7f0e',              # Orange
        'sLSTM': '#2ca02c',              # Green
        'mLSTM + sLSTM': '#d62728'       # Red
    }
    
    arch_markers = {
        'Linear Transformer': 'o',
        'mLSTM': 's',
        'sLSTM': '^',
        'mLSTM + sLSTM': 'D'
    }
    
    # Create individual plots for each key metric
    for metric in key_metrics:
        metric_df = df[df['metric_name'] == metric].copy()
        
        if metric_df.empty:
            print(f"Warning: No data found for metric {metric}")
            continue
        
        plt.figure(figsize=(8, 6))
        
        # Get unique architectures and create horizontal offsets to prevent overlap
        architectures = sorted(metric_df['architecture'].unique())
        offset_step = 0.15
        base_offsets = np.linspace(-offset_step * (len(architectures) - 1) / 2, 
                                  offset_step * (len(architectures) - 1) / 2, 
                                  len(architectures))
        
        # Plot each architecture with horizontal offset
        for i, arch in enumerate(architectures):
            arch_data = metric_df[metric_df['architecture'] == arch].sort_values('window_length')
            
            # Apply horizontal offset to x-coordinates
            x_positions = arch_data['window_length'] + base_offsets[i]
            
            plt.errorbar(
                x_positions, 
                arch_data['mean'],
                yerr=arch_data['std'],
                label=arch,
                marker=arch_markers.get(arch, 'o'),
                color=arch_colors.get(arch, 'black'),
                linewidth=2,
                markersize=8,
                capsize=4,
                capthick=1.5,
                elinewidth=2
            )
        
        plt.xlabel('Temporal Window Length (seconds)')
        plt.ylabel(metric_labels.get(metric, metric))
        plt.title(f'{metric_labels.get(metric, metric)} vs. Temporal Window Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks([4, 6, 8])  # Assuming 5s, 7s, 9s windows map to 4, 6, 8 total
        
        # Set appropriate y-axis limits based on metric
        if metric == 'balanced_accuracy':
            plt.ylim(0.45, 0.55)
        elif metric == 'cohen_kappa':
            plt.ylim(0.35, 0.55)
        elif metric == 'f1_weighted':
            plt.ylim(0.65, 0.75)
        elif metric == 'auroc_macro_ovr':
            plt.ylim(0.82, 0.89)
        elif metric == 'aucpr_macro':
            plt.ylim(0.40, 0.55)
        
        plt.tight_layout()
        
        # Save as both PNG and PDF for LaTeX compatibility
        filename_base = f'metrics_{metric}_vs_window_length'
        plt.savefig(os.path.join(output_dir, f'{filename_base}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{filename_base}.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filename_base}")
    
    # Create a comprehensive comparison plot with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        metric_df = df[df['metric_name'] == metric].copy()
        
        if metric_df.empty:
            continue
        
        # Get unique architectures and create horizontal offsets
        architectures = sorted(metric_df['architecture'].unique())
        offset_step = 0.1
        base_offsets = np.linspace(-offset_step * (len(architectures) - 1) / 2, 
                                  offset_step * (len(architectures) - 1) / 2, 
                                  len(architectures))
        
        for j, arch in enumerate(architectures):
            arch_data = metric_df[metric_df['architecture'] == arch].sort_values('window_length')
            
            # Apply horizontal offset to x-coordinates
            x_positions = arch_data['window_length'] + base_offsets[j]
            
            ax.errorbar(
                x_positions, 
                arch_data['mean'],
                yerr=arch_data['std'],
                label=arch,
                marker=arch_markers.get(arch, 'o'),
                color=arch_colors.get(arch, 'black'),
                linewidth=1.5,
                markersize=5,
                capsize=2,
                capthick=1,
                elinewidth=1.5
            )
        
        ax.set_xlabel('Window Length (s)')
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.grid(True, alpha=0.3)
        ax.set_xticks([5, 7, 9])
        
        # Set appropriate y-axis limits
        if metric == 'balanced_accuracy':
            ax.set_ylim(0.45, 0.55)
        elif metric == 'cohen_kappa':
            ax.set_ylim(0.35, 0.55)
        elif metric == 'f1_weighted':
            ax.set_ylim(0.65, 0.75)
        elif metric == 'auroc_macro_ovr':
            ax.set_ylim(0.82, 0.89)
        elif metric == 'aucpr_macro':
            ax.set_ylim(0.40, 0.55)
    
    # Remove unused subplot
    if len(key_metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    # Add overall legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.85, 0.15), ncol=1)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    
    # Save comprehensive plot
    filename_base = 'metrics_comprehensive_comparison'
    plt.savefig(os.path.join(output_dir, f'{filename_base}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{filename_base}.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive plot: {filename_base}")

def print_summary_table(df):
    """
    Print a summary table of the results.
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE - BALANCED ACCURACY BY ARCHITECTURE AND WINDOW LENGTH")
    print("="*80)
    
    # Focus on balanced accuracy for the summary
    bal_acc_df = df[df['metric_name'] == 'balanced_accuracy'].copy()
    
    if not bal_acc_df.empty:
        summary = bal_acc_df.pivot_table(
            index='architecture', 
            columns='window_length', 
            values=['mean', 'std'], 
            aggfunc='first'
        )
        
        print("\nMean ± Std (Balanced Accuracy):")
        print("-" * 60)
        
        for arch in summary.index:
            print(f"{arch:20s}", end=" ")
            for window in sorted(summary.columns.get_level_values(1).unique()):
                try:
                    mean_val = summary.loc[arch, ('mean', window)]
                    std_val = summary.loc[arch, ('std', window)]
                    print(f"{mean_val:.4f}±{std_val:.4f}", end="  ")
                except (KeyError, TypeError):
                    print("    N/A    ", end="  ")
            print()

def main():
    """
    Main function to run the analysis.
    """
    results_dir = 'results'
    
    print("Loading metrics from analysis files...")
    df = load_all_metrics(results_dir)
    
    if df.empty:
        print("Error: No data loaded. Please check the results directory and file formats.")
        return
    
    print(f"Loaded {len(df)} metric entries from {len(df['architecture'].unique())} architectures")
    print(f"Architectures found: {df['architecture'].unique().tolist()}")
    print(f"Window lengths found: {sorted(df['window_length'].unique().tolist())}")
    print(f"Metrics found: {sorted(df['metric_name'].unique().tolist())}")
    
    # Print summary table
    print_summary_table(df)
    
    # Create plots
    print("\nCreating plots...")
    create_metric_plots(df)
    
    print("\nAnalysis complete! Check the 'fig' directory for generated plots.")
    print("Generated files are suitable for inclusion in LaTeX documents.")

if __name__ == "__main__":
    main()