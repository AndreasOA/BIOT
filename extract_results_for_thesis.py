import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

def parse_folder_name(folder_name):
    """Extract configuration from folder name"""
    match = re.search(r'mlstm=(True|False)_slstm=(True|False)_secondsBeforeEvent=(\d+)_secondsAfterEvent=(\d+)_seed=(\d+)', folder_name)
    if match:
        mlstm = match.group(1) == 'True'
        slstm = match.group(2) == 'True'
        seconds_before = int(match.group(3))
        seconds_after = int(match.group(4))
        seed = int(match.group(5))
        
        # Determine architecture name
        if not mlstm and not slstm:
            arch = "Linear Transformer"
        elif mlstm and not slstm:
            arch = "mLSTM"
        elif not mlstm and slstm:
            arch = "sLSTM"
        else:
            arch = "mLSTM+sLSTM"
        
        window = f"{seconds_before}-{seconds_after}"
        total_seconds = seconds_before + 1 + seconds_after
        
        return {
            'architecture': arch,
            'mlstm': mlstm,
            'slstm': slstm,
            'window': window,
            'total_seconds': total_seconds,
            'seed': seed
        }
    return None

def parse_metrics_file(metrics_path):
    """Extract metrics from metrics.txt file"""
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract standard metrics
    metrics['accuracy'] = float(re.search(r'Accuracy:\s+(\d+\.\d+)', content).group(1))
    metrics['balanced_accuracy'] = float(re.search(r'Balanced Accuracy:\s+(\d+\.\d+)', content).group(1))
    metrics['cohen_kappa'] = float(re.search(r'Cohen Kappa:\s+(\d+\.\d+)', content).group(1))
    metrics['f1_weighted'] = float(re.search(r'F1 Weighted:\s+(\d+\.\d+)', content).group(1))
    
    # Extract ROC-AUC metrics
    metrics['auroc_macro_ovr'] = float(re.search(r'AUROC Macro \(OvR\):\s+(\d+\.\d+)', content).group(1))
    metrics['auroc_weighted_ovr'] = float(re.search(r'AUROC Weighted \(OvR\):\s+(\d+\.\d+)', content).group(1))
    metrics['auroc_macro_ovo'] = float(re.search(r'AUROC Macro \(OvO\):\s+(\d+\.\d+)', content).group(1))
    metrics['auroc_weighted_ovo'] = float(re.search(r'AUROC Weighted \(OvO\):\s+(\d+\.\d+)', content).group(1))
    
    # Extract AUC-PR metrics
    metrics['aucpr_macro'] = float(re.search(r'AUC-PR Macro:\s+(\d+\.\d+)', content).group(1))
    metrics['aucpr_micro'] = float(re.search(r'AUC-PR Micro:\s+(\d+\.\d+)', content).group(1))
    
    return metrics

def collect_all_results(base_path):
    """Collect all results from stored_runs/final_results"""
    results = []
    
    for folder in os.listdir(base_path):
        if not folder.startswith('BIOT_'):
            continue
        
        config = parse_folder_name(folder)
        if not config:
            continue
        
        folder_path = os.path.join(base_path, folder)
        
        # Try bal_acc checkpoint first
        metrics_bal_acc = os.path.join(folder_path, 'checkpoints', 'bal_acc', 'metrics.txt')
        metrics_loss = os.path.join(folder_path, 'checkpoints', 'loss', 'metrics.txt')
        
        if os.path.exists(metrics_bal_acc):
            metrics = parse_metrics_file(metrics_bal_acc)
            if metrics:
                result = {**config, **metrics, 'checkpoint_type': 'bal_acc'}
                results.append(result)
        
        if os.path.exists(metrics_loss):
            metrics = parse_metrics_file(metrics_loss)
            if metrics:
                result = {**config, **metrics, 'checkpoint_type': 'loss'}
                results.append(result)
    
    return pd.DataFrame(results)

def generate_summary_table(df, checkpoint_type='bal_acc'):
    """Generate summary statistics table grouped by architecture and window"""
    df_filtered = df[df['checkpoint_type'] == checkpoint_type]
    
    grouped = df_filtered.groupby(['architecture', 'window', 'total_seconds'])
    
    summary = []
    for (arch, window, total_sec), group in grouped:
        n_runs = len(group)
        summary.append({
            'Architecture': arch,
            'Window': window,
            'Total (s)': total_sec,
            'N': n_runs,
            'Balanced Accuracy': f"{group['balanced_accuracy'].mean():.4f} ± {group['balanced_accuracy'].std():.4f}",
            'Cohen Kappa': f"{group['cohen_kappa'].mean():.4f} ± {group['cohen_kappa'].std():.4f}",
            'F1 Weighted': f"{group['f1_weighted'].mean():.4f} ± {group['f1_weighted'].std():.4f}",
            'AUCPR Macro': f"{group['aucpr_macro'].mean():.4f} ± {group['aucpr_macro'].std():.4f}",
            'BA_mean': group['balanced_accuracy'].mean(),
            'BA_std': group['balanced_accuracy'].std()
        })
    
    return pd.DataFrame(summary).sort_values(['Architecture', 'Total (s)'])

def generate_latex_table(summary_df, caption, label):
    """Generate LaTeX table from summary dataframe"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\begin{tabular}{l | c | c | c | c | c | c}")
    latex.append("\\hline")
    latex.append("\\textbf{Architecture} & \\textbf{Window} & \\textbf{N} & \\textbf{Balanced Acc.} & \\textbf{Cohen $\\kappa$} & \\textbf{F1 Weighted} & \\textbf{AUCPR Macro} \\\\")
    latex.append("\\hline")
    
    current_arch = None
    for idx, row in summary_df.iterrows():
        if current_arch != row['Architecture']:
            if current_arch is not None:
                latex.append("\\hline")
            current_arch = row['Architecture']
        
        latex.append(f"{row['Architecture']} & {row['Window']} & {row['N']} & {row['Balanced Accuracy']} & {row['Cohen Kappa']} & {row['F1 Weighted']} & {row['AUCPR Macro']} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def parse_runtime_file(runtime_path):
    """Parse runtime.txt file"""
    runtimes = []
    
    with open(runtime_path, 'r') as f:
        lines = f.readlines()
    
    current_model = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.endswith(':'):
            current_model = line[:-1]
            continue
        
        # Parse runtime line: "6h 48m 9s - 9s - seed 42"
        match = re.search(r'(\d+)h (\d+)m (\d+)s - (\d+)s - seed (\d+)', line)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            window_sec = int(match.group(4))
            seed = int(match.group(5))
            
            total_minutes = hours * 60 + minutes + seconds / 60
            
            # Map model names
            model_map = {
                'SLSTM': 'sLSTM',
                'BIOT': 'Linear Transformer',
                'MLSTM': 'mLSTM',
                'S+MLSTM': 'mLSTM+sLSTM'
            }
            
            runtimes.append({
                'model': model_map.get(current_model, current_model),
                'window_seconds': window_sec,
                'seed': seed,
                'hours': hours,
                'minutes': minutes,
                'seconds': seconds,
                'total_minutes': total_minutes
            })
    
    return pd.DataFrame(runtimes)

def generate_runtime_table(runtime_df):
    """Generate runtime comparison table"""
    grouped = runtime_df.groupby(['model', 'window_seconds'])
    
    summary = []
    for (model, window), group in grouped:
        mean_minutes = group['total_minutes'].mean()
        std_minutes = group['total_minutes'].std()
        
        mean_hours = int(mean_minutes // 60)
        mean_mins = int(mean_minutes % 60)
        
        summary.append({
            'Model': model,
            'Window (s)': window,
            'N Runs': len(group),
            'Mean Runtime': f"{mean_hours}h {mean_mins}m",
            'Std (min)': f"{std_minutes:.1f}",
            'mean_minutes': mean_minutes
        })
    
    df = pd.DataFrame(summary).sort_values(['Model', 'Window (s)'])
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\footnotesize")
    latex.append("\\caption{Training runtime comparison across architectural variants and temporal windows. Values represent mean runtime across seeds with standard deviation.}")
    latex.append("\\label{tab:runtime_comparison}")
    latex.append("\\begin{tabular}{l | c | c | c | c}")
    latex.append("\\hline")
    latex.append("\\textbf{Architecture} & \\textbf{Window (s)} & \\textbf{N Runs} & \\textbf{Mean Runtime} & \\textbf{Std (min)} \\\\")
    latex.append("\\hline")
    
    current_model = None
    for idx, row in df.iterrows():
        if current_model != row['Model']:
            if current_model is not None:
                latex.append("\\hline")
            current_model = row['Model']
        
        latex.append(f"{row['Model']} & {row['Window (s)']} & {row['N Runs']} & {row['Mean Runtime']} & {row['Std (min)']} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

if __name__ == "__main__":
    base_path = "stored_runs/final_results"
    runtime_path = "stored_runs/final_results/runtime.txt"
    
    # Collect all results
    print("Collecting results...")
    df = collect_all_results(base_path)
    
    # Generate summary tables
    print("\nGenerating summary table for balanced accuracy checkpoint...")
    summary_bal_acc = generate_summary_table(df, 'bal_acc')
    print(summary_bal_acc.to_string())
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(
        summary_bal_acc,
        "Performance comparison across architectural variants and temporal windows using balanced accuracy checkpoints. Values represent mean ± standard deviation across multiple independent runs. Missing configurations (N<5) will be completed in future experiments.",
        "tab:metrics_bal_acc_complete"
    )
    
    # Save to file
    with open('results_table_bal_acc.tex', 'w') as f:
        f.write(latex_table)
    print("\nSaved LaTeX table to results_table_bal_acc.tex")
    
    # Parse runtime data
    if os.path.exists(runtime_path):
        print("\nParsing runtime data...")
        runtime_df = parse_runtime_file(runtime_path)
        runtime_latex = generate_runtime_table(runtime_df)
        
        with open('results_runtime_table.tex', 'w') as f:
            f.write(runtime_latex)
        print("Saved runtime table to results_runtime_table.tex")
    
    # Export complete data to CSV for further analysis
    df.to_csv('complete_results_data.csv', index=False)
    print("\nExported complete data to complete_results_data.csv")
    
    print("\nSummary statistics:")
    print(f"Total runs collected: {len(df)}")
    print(f"Architectures: {df['architecture'].unique()}")
    print(f"Windows: {df['window'].unique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
