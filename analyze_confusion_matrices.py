#!/usr/bin/env python3
"""
Confusion Matrix Analysis Script

This script processes confusion matrices from multiple model runs, performing:
1. Per-class normalization (row-wise)
2. Aggregation across seeds with uncertainty quantification
3. Statistical tests at class and cell level
4. LaTeX-ready table generation
"""

import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def find_confusion_matrix_files():
    """Find all confusion matrix CSV files in final_results."""
    base_path = Path("stored_runs/final_results")
    cm_files = []
    
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
        
        cm_path = folder / "checkpoints" / "bal_acc" / "confusion_matrix.csv"
        if cm_path.exists():
            cm_files.append((folder.name, cm_path))
    
    return sorted(cm_files)


def parse_folder_name(folder_name):
    """Extract parameters from folder name."""
    params = {}
    
    # Extract mlstm (match True or False explicitly)
    mlstm_match = re.search(r'mlstm=(True|False)', folder_name)
    if mlstm_match:
        params['mlstm'] = mlstm_match.group(1) == 'True'
    
    # Extract slstm (match True or False explicitly)
    slstm_match = re.search(r'slstm=(True|False)', folder_name)
    if slstm_match:
        params['slstm'] = slstm_match.group(1) == 'True'
    
    # Extract window parameters
    before_match = re.search(r'secondsBeforeEvent=(\d+)', folder_name)
    if before_match:
        params['seconds_before'] = int(before_match.group(1))
    
    after_match = re.search(r'secondsAfterEvent=(\d+)', folder_name)
    if after_match:
        params['seconds_after'] = int(after_match.group(1))
    
    # Extract seed
    seed_match = re.search(r'seed=(\d+)', folder_name)
    if seed_match:
        params['seed'] = int(seed_match.group(1))
    
    return params


def load_confusion_matrix(csv_path):
    """Load confusion matrix from CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    return df.values.astype(float)


def normalize_confusion_matrix(cm):
    """Normalize confusion matrix per class (row-wise)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized_cm = cm / row_sums
    return normalized_cm


def group_confusion_matrices_by_config():
    """Group confusion matrices by model configuration (excluding seed)."""
    cm_files = find_confusion_matrix_files()
    
    if not cm_files:
        print("No confusion matrix files found!")
        return {}
    
    grouped = defaultdict(list)
    
    for folder_name, cm_path in cm_files:
        params = parse_folder_name(folder_name)
        
        # Create config key (excluding seed)
        config_key = (
            params.get('mlstm', False),
            params.get('slstm', False),
            params.get('seconds_before', 0),
            params.get('seconds_after', 0)
        )
        
        # Load and normalize confusion matrix
        cm = load_confusion_matrix(cm_path)
        cm_normalized = normalize_confusion_matrix(cm)
        
        grouped[config_key].append({
            'folder_name': folder_name,
            'seed': params.get('seed', -1),
            'cm_raw': cm,
            'cm_normalized': cm_normalized
        })
    
    return grouped


def aggregate_confusion_matrices(matrices_list):
    """
    Aggregate normalized confusion matrices across seeds.
    
    Returns:
        mean_cm: Mean normalized confusion matrix
        std_cm: Standard deviation of normalized confusion matrix
        raw_matrices: List of normalized confusion matrices for statistical tests
    """
    normalized_matrices = [item['cm_normalized'] for item in matrices_list]
    
    # Stack matrices along new axis
    stacked = np.stack(normalized_matrices, axis=0)
    
    # Calculate mean and std
    mean_cm = np.mean(stacked, axis=0)
    std_cm = np.std(stacked, axis=0, ddof=1)  # Sample std
    
    return mean_cm, std_cm, normalized_matrices


def format_cell_latex(mean, std, highlight_diagonal=False, is_diagonal=False):
    """Format a confusion matrix cell for LaTeX."""
    if highlight_diagonal and is_diagonal:
        return f"\\textbf{{{mean:.3f}}}$\\pm${std:.3f}"
    else:
        return f"{mean:.3f}$\\pm${std:.3f}"


def generate_latex_table(mean_cm, std_cm, config_name, class_names=None):
    """Generate LaTeX table for confusion matrix with uncertainty."""
    n_classes = mean_cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Start LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Normalized Confusion Matrix: " + config_name + "}")
    latex.append("\\label{tab:cm_" + config_name.lower().replace(' ', '_').replace('=', '_') + "}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    
    # Table header
    col_spec = "l" + "c" * n_classes
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Column headers
    header = " & ".join(["\\textbf{True $\\backslash$ Pred}"] + [f"\\textbf{{{cn}}}" for cn in class_names])
    latex.append(header + " \\\\")
    latex.append("\\midrule")
    
    # Data rows
    for i in range(n_classes):
        row_cells = [f"\\textbf{{{class_names[i]}}}"]
        for j in range(n_classes):
            cell = format_cell_latex(mean_cm[i, j], std_cm[i, j], 
                                   highlight_diagonal=True, is_diagonal=(i==j))
            row_cells.append(cell)
        latex.append(" & ".join(row_cells) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def perform_class_level_tests(matrices_a, matrices_b, class_idx):
    """
    Perform statistical tests for a specific class (row).
    
    Tests the diagonal value (correct classification rate) for the given class.
    """
    # Extract diagonal values for the specific class across all runs
    values_a = [cm[class_idx, class_idx] for cm in matrices_a]
    values_b = [cm[class_idx, class_idx] for cm in matrices_b]
    
    # Ensure equal number of samples
    min_len = min(len(values_a), len(values_b))
    values_a = values_a[:min_len]
    values_b = values_b[:min_len]
    
    results = {}
    
    try:
        # Paired t-test
        t_stat, p_ttest = ttest_rel(values_a, values_b)
        results['t_stat'] = t_stat
        results['p_ttest'] = p_ttest
    except Exception as e:
        results['t_stat'] = np.nan
        results['p_ttest'] = np.nan
        results['t_error'] = str(e)
    
    try:
        # Wilcoxon signed-rank test
        w_stat, p_wilcoxon = wilcoxon(values_a, values_b)
        results['w_stat'] = w_stat
        results['p_wilcoxon'] = p_wilcoxon
    except Exception as e:
        results['w_stat'] = np.nan
        results['p_wilcoxon'] = np.nan
        results['w_error'] = str(e)
    
    results['mean_a'] = np.mean(values_a)
    results['mean_b'] = np.mean(values_b)
    results['std_a'] = np.std(values_a, ddof=1)
    results['std_b'] = np.std(values_b, ddof=1)
    results['values_a'] = values_a
    results['values_b'] = values_b
    
    return results


def perform_cell_level_tests(matrices_a, matrices_b, row_idx, col_idx):
    """
    Perform statistical tests for a specific confusion matrix cell.
    """
    # Extract cell values across all runs
    values_a = [cm[row_idx, col_idx] for cm in matrices_a]
    values_b = [cm[row_idx, col_idx] for cm in matrices_b]
    
    # Ensure equal number of samples
    min_len = min(len(values_a), len(values_b))
    values_a = values_a[:min_len]
    values_b = values_b[:min_len]
    
    results = {}
    
    try:
        # Paired t-test
        t_stat, p_ttest = ttest_rel(values_a, values_b)
        results['t_stat'] = t_stat
        results['p_ttest'] = p_ttest
    except Exception as e:
        results['t_stat'] = np.nan
        results['p_ttest'] = np.nan
        results['t_error'] = str(e)
    
    try:
        # Wilcoxon signed-rank test
        w_stat, p_wilcoxon = wilcoxon(values_a, values_b)
        results['w_stat'] = w_stat
        results['p_wilcoxon'] = p_wilcoxon
    except Exception as e:
        results['w_stat'] = np.nan
        results['p_wilcoxon'] = np.nan
        results['w_error'] = str(e)
    
    results['mean_a'] = np.mean(values_a)
    results['mean_b'] = np.mean(values_b)
    results['std_a'] = np.std(values_a, ddof=1)
    results['std_b'] = np.std(values_b, ddof=1)
    
    return results


def config_to_name(config):
    """Convert config tuple to readable name."""
    mlstm, slstm, before, after = config
    
    model_type = ""
    if mlstm and slstm:
        model_type = "mLSTM+sLSTM"
    elif mlstm and not slstm:
        model_type = "mLSTM"
    elif not mlstm and slstm:
        model_type = "sLSTM"
    else:
        model_type = "Linear Transformer"
    
    return f"{model_type} (±{before}s/±{after}s)"


def plot_confusion_matrix_heatmap(mean_cm, std_cm, config_name, class_names=None, 
                                  output_path=None):
    """Create a heatmap visualization of the confusion matrix."""
    n_classes = mean_cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotations with mean ± std
    annot = np.empty_like(mean_cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot[i, j] = f"{mean_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}"
    
    # Create heatmap
    sns.heatmap(mean_cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'},
                vmin=0, vmax=1, ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(f'Normalized Confusion Matrix\n{config_name}', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {output_path}")
    
    return fig


def generate_comparison_latex_table(config_a, config_b, matrices_a, matrices_b, 
                                   class_names=None, test_type='class'):
    """
    Generate LaTeX table comparing two configurations.
    
    Args:
        test_type: 'class' for diagonal only, 'full' for all cells
    """
    n_classes = matrices_a[0].shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    config_a_name = config_to_name(config_a)
    config_b_name = config_to_name(config_b)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Statistical Comparison: {config_a_name} vs {config_b_name}}}")
    
    if test_type == 'class':
        # Class-level comparison (diagonal only)
        latex.append("\\label{tab:comparison_class_" + 
                    config_a_name.lower().replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','') + 
                    "_vs_" +
                    config_b_name.lower().replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','') + "}")
        
        latex.append("\\begin{tabular}{lccccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Class} & \\textbf{Model A} & \\textbf{Model B} & \\textbf{t-stat} & \\textbf{p-value (t)} & \\textbf{p-value (W)} \\\\")
        latex.append("\\midrule")
        
        for i in range(n_classes):
            results = perform_class_level_tests(matrices_a, matrices_b, i)
            
            sig_t = "**" if results['p_ttest'] < 0.01 else "*" if results['p_ttest'] < 0.05 else ""
            sig_w = "**" if results['p_wilcoxon'] < 0.01 else "*" if results['p_wilcoxon'] < 0.05 else ""
            
            row = (f"{class_names[i]} & "
                  f"{results['mean_a']:.3f}$\\pm${results['std_a']:.3f} & "
                  f"{results['mean_b']:.3f}$\\pm${results['std_b']:.3f} & "
                  f"{results['t_stat']:.2f} & "
                  f"{results['p_ttest']:.3f}{sig_t} & "
                  f"{results['p_wilcoxon']:.3f}{sig_w} \\\\")
            latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\multicolumn{6}{l}{\\textit{* p < 0.05, ** p < 0.01}} \\\\")
        latex.append("\\end{tabular}")
    
    else:  # full cell-level comparison
        latex.append("\\label{tab:comparison_full_" + 
                    config_a_name.lower().replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','') + 
                    "_vs_" +
                    config_b_name.lower().replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','') + "}")
        
        latex.append("\\resizebox{\\textwidth}{!}{")
        latex.append("\\begin{tabular}{lllcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{True} & \\textbf{Pred} & \\textbf{Cell} & \\textbf{Model A} & \\textbf{Model B} & \\textbf{t-stat} & \\textbf{p (t)} \\\\")
        latex.append("\\midrule")
        
        for i in range(n_classes):
            for j in range(n_classes):
                results = perform_cell_level_tests(matrices_a, matrices_b, i, j)
                
                sig_t = "**" if results['p_ttest'] < 0.01 else "*" if results['p_ttest'] < 0.05 else ""
                
                cell_type = "Diag." if i == j else "Off"
                
                row = (f"{class_names[i]} & {class_names[j]} & {cell_type} & "
                      f"{results['mean_a']:.3f}$\\pm${results['std_a']:.3f} & "
                      f"{results['mean_b']:.3f}$\\pm${results['std_b']:.3f} & "
                      f"{results['t_stat']:.2f} & "
                      f"{results['p_ttest']:.3f}{sig_t} \\\\")
                latex.append(row)
        
        latex.append("\\bottomrule")
        latex.append("\\multicolumn{7}{l}{\\textit{* p < 0.05, ** p < 0.01}} \\\\")
        latex.append("\\end{tabular}")
        latex.append("}")
    
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    
    # Load and group confusion matrices
    print("\n📊 Loading confusion matrices...")
    grouped_cms = group_confusion_matrices_by_config()
    
    if not grouped_cms:
        print("❌ No confusion matrices found!")
        return
    
    print(f"✓ Found {len(grouped_cms)} unique configurations")
    
    # Display configurations
    print("\n" + "="*80)
    print("AVAILABLE CONFIGURATIONS")
    print("="*80)
    configs = list(grouped_cms.keys())
    for idx, config in enumerate(configs, 1):
        name = config_to_name(config)
        n_seeds = len(grouped_cms[config])
        seeds = [item['seed'] for item in grouped_cms[config]]
        print(f"{idx:2d}. {name:40s} ({n_seeds} seeds: {seeds})")
    
    # Create output directory
    output_dir = Path("confusion_matrix_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Process each configuration
    print("\n" + "="*80)
    print("PROCESSING CONFIGURATIONS")
    print("="*80)
    
    aggregated_results = {}
    
    for config, matrices_list in grouped_cms.items():
        config_name = config_to_name(config)
        print(f"\n📈 Processing: {config_name}")
        print(f"   Number of seeds: {len(matrices_list)}")
        
        # Aggregate matrices
        mean_cm, std_cm, normalized_matrices = aggregate_confusion_matrices(matrices_list)
        
        aggregated_results[config] = {
            'name': config_name,
            'mean_cm': mean_cm,
            'std_cm': std_cm,
            'normalized_matrices': normalized_matrices,
            'n_seeds': len(matrices_list)
        }
        
        # Generate LaTeX table
        latex_table = generate_latex_table(mean_cm, std_cm, config_name)
        
        # Save LaTeX table
        safe_name = config_name.replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','_')
        latex_path = output_dir / f"cm_table_{safe_name}.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"   ✓ Saved LaTeX table: {latex_path}")
        
        # Generate heatmap
        plot_path = output_dir / f"cm_heatmap_{safe_name}.pdf"
        plot_confusion_matrix_heatmap(mean_cm, std_cm, config_name, output_path=plot_path)
        plt.close()
        
        # Print summary statistics
        print(f"   Mean diagonal accuracy: {np.mean(np.diag(mean_cm)):.3f} ± {np.mean(np.diag(std_cm)):.3f}")
    
    # Statistical comparisons - automatically compare all pairs
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)
    print("\nGenerating all pairwise comparisons...")
    
    configs = list(aggregated_results.keys())
    comparison_count = 0
    
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            config_a = configs[i]
            config_b = configs[j]
            
            print(f"\n🔬 Comparing: {aggregated_results[config_a]['name']} vs {aggregated_results[config_b]['name']}")
            
            # Class-level comparison
            latex_class = generate_comparison_latex_table(
                config_a, config_b,
                aggregated_results[config_a]['normalized_matrices'],
                aggregated_results[config_b]['normalized_matrices'],
                test_type='class'
            )
            
            safe_name_a = aggregated_results[config_a]['name'].replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','_')
            safe_name_b = aggregated_results[config_b]['name'].replace(' ', '_').replace('(','').replace(')','').replace('/','_').replace('+','_')
            
            comparison_path = output_dir / f"comparison_class_{safe_name_a}_vs_{safe_name_b}.tex"
            with open(comparison_path, 'w') as f:
                f.write(latex_class)
            print(f"   ✓ Saved comparison: {comparison_path}")
            
            comparison_count += 1
    
    print(f"\n✓ Generated {comparison_count} pairwise comparisons")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - LaTeX tables for each configuration")
    print("  - PDF heatmaps for each configuration")
    print("  - Statistical comparison tables (if requested)")


if __name__ == "__main__":
    main()
