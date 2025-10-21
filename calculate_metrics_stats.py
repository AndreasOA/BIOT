import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import glob
from scipy.stats import ttest_rel, wilcoxon

def find_run_folders():
    """Find all run folders in stored_runs directory."""
    stored_runs_path = "stored_runs/final_results"
    if not os.path.exists(stored_runs_path):
        print(f"Error: {stored_runs_path} directory not found!")
        return []
    
    # Get all folders in stored_runs
    folders = [f for f in os.listdir(stored_runs_path) 
              if os.path.isdir(os.path.join(stored_runs_path, f))]
    
    # Filter folders that have metrics files
    valid_folders = []
    for folder in folders:
        bal_acc_path = os.path.join(stored_runs_path, folder, "checkpoints", "bal_acc", "metrics.txt")
        loss_path = os.path.join(stored_runs_path, folder, "checkpoints", "loss", "metrics.txt")
        if os.path.exists(bal_acc_path) or os.path.exists(loss_path):
            valid_folders.append(folder)
    
    return valid_folders

def display_folder_selection(folders):
    """Display available folders and let user select which ones to analyze."""
    print("=" * 120)
    print("AVAILABLE RUN FOLDERS")
    print("=" * 120)
    
    for i, folder_name in enumerate(folders, 1):
        print(f"{i:2d}. {folder_name}")
    
    print("\n" + "=" * 120)
    print("SELECTION OPTIONS:")
    print("  - Enter folder numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'all' to select all folders")
    print("  - Enter 'q' to quit")
    print("=" * 120)
    
    while True:
        selection = input("\nEnter your selection: ").strip().lower()
        
        if selection == 'q':
            return []
        elif selection == 'all':
            return folders
        else:
            try:
                # Parse comma-separated numbers
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_folders = []
                
                for idx in indices:
                    if 0 <= idx < len(folders):
                        selected_folders.append(folders[idx])
                    else:
                        print(f"Warning: Invalid index {idx + 1}")
                
                if selected_folders:
                    return selected_folders
                else:
                    print("No valid folders selected. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas, 'all', or 'q'.")

def display_group_selection(folders, group_name):
    """Display available folders and let user select a group for statistical comparison."""
    print(f"\n{'=' * 120}")
    print(f"SELECT FOLDERS FOR {group_name.upper()}")
    print("=" * 120)
    
    for i, folder_name in enumerate(folders, 1):
        print(f"{i:2d}. {folder_name}")
    
    print("\n" + "=" * 120)
    print("SELECTION OPTIONS:")
    print("  - Enter folder numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'q' to quit")
    print("=" * 120)
    
    while True:
        selection = input(f"\nEnter your selection for {group_name}: ").strip().lower()
        
        if selection == 'q':
            return []
        else:
            try:
                # Parse comma-separated numbers
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_folders = []
                
                for idx in indices:
                    if 0 <= idx < len(folders):
                        selected_folders.append(folders[idx])
                    else:
                        print(f"Warning: Invalid index {idx + 1}")
                
                if selected_folders:
                    return selected_folders
                else:
                    print("No valid folders selected. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas or 'q'.")

def get_analysis_type():
    """Ask user what type of analysis to perform."""
    print("\n" + "=" * 50)
    print("SELECT ANALYSIS TYPE")
    print("=" * 50)
    print("1. Statistics only (mean, std, CV)")
    print("2. Statistical comparison (Wilcoxon & t-test)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1':
            return 'stats_only'
        elif choice == '2':
            return 'comparison'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def get_metrics_type():
    """Ask user which metrics to analyze: balanced accuracy or loss."""
    print("\n" + "=" * 50)
    print("SELECT METRICS TYPE")
    print("=" * 50)
    print("1. Balanced Accuracy")
    print("2. Loss")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1':
            return 'bal_acc'
        elif choice == '2':
            return 'loss'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def extract_metrics_from_file(file_path):
    """Extract metrics from a metrics.txt file."""
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract standard metrics
        accuracy_match = re.search(r'Accuracy:\s+(\d+\.\d+)', content)
        if accuracy_match:
            metrics['accuracy'] = float(accuracy_match.group(1))
        
        bal_acc_match = re.search(r'Balanced Accuracy:\s+(\d+\.\d+)', content)
        if bal_acc_match:
            metrics['balanced_accuracy'] = float(bal_acc_match.group(1))
        
        cohen_match = re.search(r'Cohen Kappa:\s+(\d+\.\d+)', content)
        if cohen_match:
            metrics['cohen_kappa'] = float(cohen_match.group(1))
        
        f1_match = re.search(r'F1 Weighted:\s+(\d+\.\d+)', content)
        if f1_match:
            metrics['f1_weighted'] = float(f1_match.group(1))
        
        # Extract ROC-AUC metrics
        auroc_macro_ovr_match = re.search(r'AUROC Macro \(OvR\):\s+(\d+\.\d+)', content)
        if auroc_macro_ovr_match:
            metrics['auroc_macro_ovr'] = float(auroc_macro_ovr_match.group(1))
        
        auroc_weighted_ovr_match = re.search(r'AUROC Weighted \(OvR\):\s+(\d+\.\d+)', content)
        if auroc_weighted_ovr_match:
            metrics['auroc_weighted_ovr'] = float(auroc_weighted_ovr_match.group(1))
        
        auroc_macro_ovo_match = re.search(r'AUROC Macro \(OvO\):\s+(\d+\.\d+)', content)
        if auroc_macro_ovo_match:
            metrics['auroc_macro_ovo'] = float(auroc_macro_ovo_match.group(1))
        
        auroc_weighted_ovo_match = re.search(r'AUROC Weighted \(OvO\):\s+(\d+\.\d+)', content)
        if auroc_weighted_ovo_match:
            metrics['auroc_weighted_ovo'] = float(auroc_weighted_ovo_match.group(1))
        
        # Extract AUC-PR metrics
        aucpr_macro_match = re.search(r'AUC-PR Macro:\s+(\d+\.\d+)', content)
        if aucpr_macro_match:
            metrics['aucpr_macro'] = float(aucpr_macro_match.group(1))
        
        aucpr_micro_match = re.search(r'AUC-PR Micro:\s+(\d+\.\d+)', content)
        if aucpr_micro_match:
            metrics['aucpr_micro'] = float(aucpr_micro_match.group(1))
        
        return metrics
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

def get_common_prefix(folders):
    """Extract common prefix from folder names up to '_seed'."""
    if not folders:
        return "unknown"
    
    # Find the common prefix among all folder names
    common_prefix = folders[0]
    for folder in folders[1:]:
        # Find common characters from the start
        i = 0
        while i < min(len(common_prefix), len(folder)) and common_prefix[i] == folder[i]:
            i += 1
        common_prefix = common_prefix[:i]
    
    # Truncate at '_seed' if present
    if '_seed' in common_prefix:
        common_prefix = common_prefix[:common_prefix.find('_seed')]
    
    # Remove trailing underscores or hyphens
    common_prefix = common_prefix.rstrip('_-')
    
    return common_prefix if common_prefix else "unknown"

def calculate_metrics_statistics(selected_folders, metrics_type):
    """Calculate mean and standard deviation for metrics across selected folders."""
    
    # Collect metrics from all selected folders
    all_metrics = {}
    for folder in selected_folders:
        metrics_path = os.path.join("stored_runs/final_results", folder, "checkpoints", metrics_type, "metrics.txt")
        
        if not os.path.exists(metrics_path):
            print(f"✗ Warning: No {metrics_type} metrics found in {folder}")
            continue
            
        metrics = extract_metrics_from_file(metrics_path)
        if metrics:
            all_metrics[folder] = metrics
            print(f"✓ Successfully read metrics from: {folder}")
        else:
            print(f"✗ Failed to read metrics from: {folder}")
    
    if not all_metrics:
        print("No valid metrics files found!")
        return None
    
    # Convert to DataFrame where each row is a run (folder)
    df = pd.DataFrame(all_metrics).T
    
    # Calculate statistics across all runs
    results = {}
    for metric in df.columns:
        values = df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        results[metric] = {'mean': mean_val, 'std': std_val, 'cv': cv}
    
    # Print results
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS ACROSS {len(all_metrics)} RUNS")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'CV (%)':<10}")
    print("-" * 70)
    
    for metric, stats in results.items():
        print(f"{metric:<25} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['cv']:<10.2f}")
    
    print("=" * 70)
    
    # Print individual run values
    print("\nINDIVIDUAL RUN VALUES:")
    print("-" * 70)
    for metric in df.columns:
        print(f"\n{metric}:")
        for folder, value in df[metric].items():
            print(f"  {folder}: {value:.4f}")
    
    return {'results': results, 'dataframe': df}

def collect_metrics_from_folders(folders, metrics_type):
    """Collect metrics from specified folders and return as DataFrame."""
    all_metrics = {}
    
    for folder in folders:
        metrics_path = os.path.join("stored_runs/final_results", folder, "checkpoints", metrics_type, "metrics.txt")
        
        if not os.path.exists(metrics_path):
            print(f"✗ Warning: No {metrics_type} metrics found in {folder}")
            continue
            
        metrics = extract_metrics_from_file(metrics_path)
        if metrics:
            all_metrics[folder] = metrics
            print(f"✓ Successfully read metrics from: {folder}")
        else:
            print(f"✗ Failed to read metrics from: {folder}")
    
    if not all_metrics:
        return None
    
    return pd.DataFrame(all_metrics).T

def perform_statistical_tests(group_a_df, group_b_df, group_a_name="Model A", group_b_name="Model B"):
    """Perform paired t-test and Wilcoxon signed-rank test between two groups."""
    results = {}
    
    # Get common metrics between both groups
    common_metrics = set(group_a_df.columns) & set(group_b_df.columns)
    
    if not common_metrics:
        print("No common metrics found between the two groups!")
        return None
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL TESTS: {group_a_name} vs {group_b_name}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'t-test t':<12} {'t-test p':<12} {'Wilcoxon W':<12} {'Wilcoxon p':<12}")
    print("-" * 80)
    
    for metric in sorted(common_metrics):
        group_a_values = group_a_df[metric].values
        group_b_values = group_b_df[metric].values
        
        # Ensure same number of runs for paired tests
        min_length = min(len(group_a_values), len(group_b_values))
        if len(group_a_values) != len(group_b_values):
            print(f"Warning: Different number of runs for {metric}. Using first {min_length} runs.")
            group_a_values = group_a_values[:min_length]
            group_b_values = group_b_values[:min_length]
        
        # Paired t-test
        try:
            t_stat, p_ttest = ttest_rel(group_a_values, group_b_values)
        except Exception as e:
            print(f"Error in t-test for {metric}: {e}")
            t_stat, p_ttest = np.nan, np.nan
        
        # Wilcoxon signed-rank test
        try:
            w_stat, p_wilcoxon = wilcoxon(group_a_values, group_b_values)
        except Exception as e:
            print(f"Error in Wilcoxon test for {metric}: {e}")
            w_stat, p_wilcoxon = np.nan, np.nan
        
        results[metric] = {
            't_stat': t_stat,
            'p_ttest': p_ttest,
            'w_stat': w_stat,
            'p_wilcoxon': p_wilcoxon,
            'group_a_values': group_a_values,
            'group_b_values': group_b_values,
            'group_a_mean': np.mean(group_a_values),
            'group_b_mean': np.mean(group_b_values)
        }
        
        print(f"{metric:<25} {t_stat:<12.3f} {p_ttest:<12.3f} {w_stat:<12.0f} {p_wilcoxon:<12.3f}")
    
    print("=" * 80)
    
    return results

def save_results_to_file(results, metrics_type, selected_folders, filename=None):
    """Save results to a text file."""
    if not results:
        return
    
    if filename is None:
        common_prefix = get_common_prefix(selected_folders)
        filename = f"metrics_{common_prefix}_{metrics_type}_analysis.txt"
    
    with open(filename, 'w') as f:
        f.write("METRICS ANALYSIS RESULTS\n")
        f.write(f"Metrics Type: {metrics_type}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"OVERALL STATISTICS ACROSS {len(results['dataframe'])} RUNS\n")
        f.write("-" * 50 + "\n")
        
        f.write(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'CV (%)':<10}\n")
        f.write("-" * 70 + "\n")
        
        for metric, stats in results['results'].items():
            f.write(f"{metric:<25} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['cv']:<10.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        
        # Write individual run values
        f.write("\nINDIVIDUAL RUN VALUES:\n")
        f.write("-" * 70 + "\n")
        
        df = results['dataframe']
        for metric in df.columns:
            f.write(f"\n{metric}:\n")
            for folder, value in df[metric].items():
                f.write(f"  {folder}: {value:.4f}\n")
    
    print(f"\nResults saved to: {filename}")

def save_statistical_test_results(test_results, metrics_type, group_a_name, group_b_name, group_a_folders, group_b_folders, filename=None):
    """Save statistical test results to a text file."""
    if not test_results:
        return
    
    if filename is None:
        common_prefix_a = get_common_prefix(group_a_folders)
        common_prefix_b = get_common_prefix(group_b_folders)
        filename = f"metrics_{common_prefix_a}_vs_{common_prefix_b}_{metrics_type}_comparison.txt"
    
    with open(filename, 'w') as f:
        f.write("STATISTICAL TEST RESULTS\n")
        f.write(f"Metrics Type: {metrics_type}\n")
        f.write(f"Group A ({group_a_name}) vs Group B ({group_b_name})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Metric':<25} {'t-test t':<12} {'t-test p':<12} {'Wilcoxon W':<12} {'Wilcoxon p':<12}\n")
        f.write("-" * 80 + "\n")
        
        for metric, stats in test_results.items():
            f.write(f"{metric:<25} {stats['t_stat']:<12.3f} {stats['p_ttest']:<12.3f} {stats['w_stat']:<12.0f} {stats['p_wilcoxon']:<12.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # Write detailed comparison
        f.write("\nDETAILED COMPARISON:\n")
        f.write("-" * 80 + "\n")
        
        for metric, stats in test_results.items():
            f.write(f"\n{metric}:\n")
            f.write(f"  {group_a_name} mean: {stats['group_a_mean']:.4f}\n")
            f.write(f"  {group_b_name} mean: {stats['group_b_mean']:.4f}\n")
            f.write(f"  Difference: {stats['group_a_mean'] - stats['group_b_mean']:.4f}\n")
            f.write(f"  {group_a_name} values: {[f'{v:.4f}' for v in stats['group_a_values']]}\n")
            f.write(f"  {group_b_name} values: {[f'{v:.4f}' for v in stats['group_b_values']]}\n")
            
            # Significance indicators
            significance_ttest = "**" if stats['p_ttest'] < 0.01 else "*" if stats['p_ttest'] < 0.05 else ""
            significance_wilcoxon = "**" if stats['p_wilcoxon'] < 0.01 else "*" if stats['p_wilcoxon'] < 0.05 else ""
            
            f.write(f"  t-test significance: p={stats['p_ttest']:.3f} {significance_ttest}\n")
            f.write(f"  Wilcoxon significance: p={stats['p_wilcoxon']:.3f} {significance_wilcoxon}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Significance levels: * p < 0.05, ** p < 0.01\n")
    
    print(f"\nStatistical test results saved to: {filename}")

if __name__ == "__main__":
    print("🔍 Discovering run folders in stored_runs directory...")
    
    # Find all run folders
    folders = find_run_folders()
    
    if not folders:
        print("No valid run folders found in stored_runs directory!")
        print("Make sure you have metrics.txt files in the expected locations:")
        print("  stored_runs/RUN_FOLDER/checkpoints/bal_acc/metrics.txt")
        print("  stored_runs/RUN_FOLDER/checkpoints/loss/metrics.txt")
        exit(1)
    
    # Ask what type of analysis to perform
    analysis_type = get_analysis_type()
    
    if analysis_type == 'stats_only':
        # Original workflow - select folders and calculate statistics
        selected_folders = display_folder_selection(folders)
        
        if not selected_folders:
            print("No folders selected. Exiting.")
            exit(0)
        
        # Ask which metrics to analyze
        metrics_type = get_metrics_type()
        
        # Calculate statistics
        results = calculate_metrics_statistics(selected_folders, metrics_type)
        
        if results:
            # Ask if user wants to save results
            save_choice = input("\nSave results to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                save_results_to_file(results, metrics_type, selected_folders)
    
    elif analysis_type == 'comparison':
        # New workflow - select two groups and perform statistical tests
        
        # Ask which metrics to analyze
        metrics_type = get_metrics_type()
        
        # Select Group A folders
        print("\n" + "🔵" * 40)
        print("STEP 1: Select folders for Model A")
        print("🔵" * 40)
        group_a_folders = display_group_selection(folders, "Model A")
        
        if not group_a_folders:
            print("No folders selected for Model A. Exiting.")
            exit(0)
        
        # Select Group B folders
        print("\n" + "🟢" * 40)
        print("STEP 2: Select folders for Model B")
        print("🟢" * 40)
        group_b_folders = display_group_selection(folders, "Model B")
        
        if not group_b_folders:
            print("No folders selected for Model B. Exiting.")
            exit(0)
        
        # Collect metrics from both groups
        print(f"\n📊 Collecting metrics from Model A folders ({metrics_type})...")
        group_a_df = collect_metrics_from_folders(group_a_folders, metrics_type)
        
        print(f"\n📊 Collecting metrics from Model B folders ({metrics_type})...")
        group_b_df = collect_metrics_from_folders(group_b_folders, metrics_type)
        
        if group_a_df is None or group_b_df is None:
            print("Failed to collect metrics from one or both groups!")
            exit(1)
        
        # Perform statistical tests
        test_results = perform_statistical_tests(group_a_df, group_b_df, "Model A", "Model B")
        
        if test_results:
            # Display summary statistics for both groups
            print(f"\n{'='*80}")
            print("SUMMARY STATISTICS")
            print("=" * 80)
            
            print("\nModel A Statistics:")
            print("-" * 40)
            for metric in group_a_df.columns:
                mean_val = np.mean(group_a_df[metric])
                std_val = np.std(group_a_df[metric], ddof=1)
                print(f"{metric:<25} {mean_val:<12.4f} ± {std_val:<8.4f}")
            
            print("\nModel B Statistics:")
            print("-" * 40)
            for metric in group_b_df.columns:
                mean_val = np.mean(group_b_df[metric])
                std_val = np.std(group_b_df[metric], ddof=1)
                print(f"{metric:<25} {mean_val:<12.4f} ± {std_val:<8.4f}")
            
            # Ask if user wants to save results
            save_choice = input("\nSave statistical test results to file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                save_statistical_test_results(test_results, metrics_type, "Model A", "Model B", group_a_folders, group_b_folders)
    
    print("\n✅ Analysis complete!") 