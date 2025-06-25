import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import glob

def find_run_folders():
    """Find all run folders in stored_runs directory."""
    stored_runs_path = "stored_runs"
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

def calculate_metrics_statistics(selected_folders, metrics_type):
    """Calculate mean and standard deviation for metrics across selected folders."""
    
    # Collect metrics from all selected folders
    all_metrics = {}
    for folder in selected_folders:
        metrics_path = os.path.join("stored_runs", folder, "checkpoints", metrics_type, "metrics.txt")
        
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

def save_results_to_file(results, metrics_type, filename="metrics_analysis_results.txt"):
    """Save results to a text file."""
    if not results:
        return
    
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
    
    # Let user select folders
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
            save_results_to_file(results, metrics_type)
    
    print("\n✅ Analysis complete!") 