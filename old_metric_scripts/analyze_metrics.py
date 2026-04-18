import pandas as pd
import numpy as np
import glob
import re
from collections import defaultdict

def get_model_type(model_name):
    # Remove '3*' prefix and run suffix (_1, _2, _3) if they exist
    model_name = re.sub(r'^3\*', '', model_name)
    model_name = re.sub(r'_\d+$', '', model_name)
    if 'biot' in model_name:
        return 'Linear\\newline Transformer'
    elif 'm+slstm' in model_name:
        return 'M+S-LSTM'
    elif 'mlstm' in model_name:
        return 'M-LSTM'
    elif 'slstm' in model_name:
        return 'S-LSTM'
    return model_name

def get_sample_length(model_name):
    match = re.search(r'(\d+)sec', model_name)
    return int(match.group(1)) if match else None

def clean_metric_name(metric_name):
    # Remove 'test/epoch_' prefix if it exists and replace underscores with spaces
    metric_name = re.sub(r'^test/epoch_', '', metric_name)
    return metric_name.replace('_', ' ')

def analyze_metrics_simplified(folder_path):
    csv_files = glob.glob(f'{folder_path}/*.csv')
    # Nested dict: model_type -> sample_length -> metric -> list of runs
    all_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file in csv_files:
        df = pd.read_csv(file)
        for column in df.columns:
            if column == 'Step':
                continue
            match = re.match(r'(.*?)\s*-\s*(.*?)(?:__MIN|__MAX)?$', column)
            if match and not column.endswith('__MIN') and not column.endswith('__MAX'):
                model_name, metric_name = match.groups()
                model_type = get_model_type(model_name)
                sample_length = get_sample_length(model_name)
                if sample_length is None:
                    continue

                values = pd.to_numeric(df[column].dropna(), errors='coerce')
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    continue

                stat = {
                    'mean': np.mean(values),
                    'var': np.var(values)
                }
                all_stats[model_type][sample_length][clean_metric_name(metric_name)].append(stat)

    # Get all unique metrics
    all_metrics = set()
    for model_type in all_stats:
        for sample_length in all_stats[model_type]:
            all_metrics.update(all_stats[model_type][sample_length].keys())
    all_metrics = sorted(all_metrics)

    # Prepare LaTeX table
    latex = []
    latex.append(r"\begin{tabular}{l | l" + " | r" * len(all_metrics) + "}")
    
    # Header row
    header = ["Model", "\\makecell{Sample \\\\ Length}"]
    header.extend(all_metrics)
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\hline")

    # Model types in desired order
    model_types = ['Linear\\newline Transformer', 'M+S-LSTM', 'M-LSTM', 'S-LSTM']
    sample_lengths = [5, 7, 9]

    # Find highest values for each metric across all models and sample lengths
    highest_values = defaultdict(lambda: float('-inf'))
    for model_type in model_types:
        for sample_length in sample_lengths:
            for metric in all_metrics:
                runs = all_stats[model_type][sample_length][metric]
                if runs:
                    means = [r['mean'] for r in runs]
                    mean_val = np.mean(means)
                    highest_values[metric] = max(highest_values[metric], mean_val)

    for model_type in model_types:
        first_row = True
        for sample_length in sample_lengths:
            row = []
            if first_row:
                row.append(f"\\multirow{{3}}{{*}}{{{model_type}}}")
                first_row = False
            else:
                row.append("")
            row.append(f"{sample_length}s")

            for metric in all_metrics:
                runs = all_stats[model_type][sample_length][metric]
                if runs:
                    means = [r['mean'] for r in runs]
                    mean_val = np.mean(means)
                    mean_var = np.var(means)
                    
                    # Format mean value with bold if it's the highest
                    mean_str = f"\\textbf{{{mean_val:.3f}}} ±{np.sqrt(mean_var):.3f}" if abs(mean_val - highest_values[metric]) < 1e-6 else f"{mean_val:.3f} ±{np.sqrt(mean_var):.3f}"
                    row.append(mean_str)
                else:
                    row.append("-")

            latex.append(" & ".join(row) + r" \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")

    output_file = f'{folder_path}_table.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    print(f"LaTeX table written to {output_file}")

def analyze_metrics():
    # Original analysis for wandb_export folder
    csv_files = glob.glob('wandb_export/*.csv')
    # Nested dict: model_type -> sample_length -> metric -> list of runs
    all_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file in csv_files:
        df = pd.read_csv(file)
        for column in df.columns:
            if column == 'Step':
                continue
            match = re.match(r'(.*?)\s*-\s*(.*?)(?:__MIN|__MAX)?$', column)
            if match and not column.endswith('__MIN') and not column.endswith('__MAX'):
                model_name, metric_name = match.groups()
                model_type = get_model_type(model_name)
                sample_length = get_sample_length(model_name)
                if sample_length is None:
                    continue

                values = pd.to_numeric(df[column].dropna(), errors='coerce')
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    continue

                stat = {
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'var': np.var(values)
                }
                all_stats[model_type][sample_length][clean_metric_name(metric_name)].append(stat)

    # Get all unique metrics
    all_metrics = set()
    for model_type in all_stats:
        for sample_length in all_stats[model_type]:
            all_metrics.update(all_stats[model_type][sample_length].keys())
    all_metrics = sorted(all_metrics)

    # Prepare LaTeX table
    latex = []
    # Add vertical lines between model, sample length, and each metric group
    latex.append(r"\begin{tabular}{l | l" + " | r r" * len(all_metrics) + "}")
    
    # Header row 1: Metrics
    header1 = ["Model", "\\makecell{Sample \\\\ Length}"]
    for metric in all_metrics:
        header1.extend([f"\\multicolumn{{2}}{{c}}{{{metric}}}"])
    latex.append(" & ".join(header1) + r" \\")
    
    # Header row 2: Statistics
    header2 = ["", ""]
    for _ in all_metrics:
        header2.extend(["Max", "Mean"])
    latex.append(" & ".join(header2) + r" \\")
    latex.append(r"\hline")

    # Model types in desired order
    model_types = ['Linear\\newline Transformer', 'M+S-LSTM', 'M-LSTM', 'S-LSTM']
    sample_lengths = [5, 7, 9]

    # Find highest values for each metric across all models and sample lengths
    highest_values = defaultdict(lambda: {'max': float('-inf'), 'mean': float('-inf')})
    for model_type in model_types:
        for sample_length in sample_lengths:
            for metric in all_metrics:
                runs = all_stats[model_type][sample_length][metric]
                if runs:
                    maxs = [r['max'] for r in runs]
                    means = [r['mean'] for r in runs]
                    max_val = np.mean(maxs)
                    mean_val = np.mean(means)
                    highest_values[metric]['max'] = max(highest_values[metric]['max'], max_val)
                    highest_values[metric]['mean'] = max(highest_values[metric]['mean'], mean_val)

    for model_type in model_types:
        first_row = True
        for sample_length in sample_lengths:
            row = []
            if first_row:
                row.append(f"\\multirow{{3}}{{*}}{{{model_type}}}")
                first_row = False
            else:
                row.append("")
            row.append(f"{sample_length}s")

            for metric in all_metrics:
                runs = all_stats[model_type][sample_length][metric]
                if runs:
                    maxs = [r['max'] for r in runs]
                    means = [r['mean'] for r in runs]
                    max_val = np.mean(maxs)
                    max_var = np.var(maxs)
                    mean_val = np.mean(means)
                    mean_var = np.var(means)
                    
                    # Format max value with bold if it's the highest
                    max_str = f"\\textbf{{{max_val:.3f}}} ±{np.sqrt(max_var):.3f}" if abs(max_val - highest_values[metric]['max']) < 1e-6 else f"{max_val:.3f} ±{np.sqrt(max_var):.3f}"
                    # Format mean value with bold if it's the highest
                    mean_str = f"\\textbf{{{mean_val:.3f}}} ±{np.sqrt(mean_var):.3f}" if abs(mean_val - highest_values[metric]['mean']) < 1e-6 else f"{mean_val:.3f} ±{np.sqrt(mean_var):.3f}"
                    
                    row.extend([max_str, mean_str])
                else:
                    row.extend(["-", "-"])

            latex.append(" & ".join(row) + r" \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")

    with open('metrics_table.tex', 'w') as f:
        f.write('\n'.join(latex))
    print("LaTeX table written to metrics_table.tex")

if __name__ == "__main__":
    #analyze_metrics()  # Original analysis for wandb_export
    analyze_metrics_simplified('wandb_export_2')  # Simplified analysis for wandb_export_2 