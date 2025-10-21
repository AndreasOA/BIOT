import pandas as pd
import numpy as np
import re
import os
import glob
from collections import defaultdict

def extract_model_info(column_name):
    """Extract model type and variant from column name"""
    # Remove the metric suffix
    model_name = column_name.split(' - ')[0]
    
    # Extract base model type (remove default_X suffix)
    base_model = re.sub(r'_default_[123]$', '', model_name)
    
    # Extract variant (default_1, default_2, default_3)
    variant_match = re.search(r'default_([123])$', model_name)
    variant = variant_match.group(1) if variant_match else None
    
    return base_model, variant

def extract_metric_name(filename):
    """Extract metric name from filename"""
    # Remove path and extension
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    
    # Remove wandb_export_ prefix
    if basename.startswith('wandb_export_'):
        metric_name = basename[13:]  # Remove 'wandb_export_' prefix
        return metric_name
    return basename

def main():
    # Read validation CSV file
    val_df = pd.read_csv('wandb_export_3/wandb_loss_val.csv')
    
    # Get all CSV files in wandb_export directory
    test_files = glob.glob('wandb_export/*.csv')
    
    # Remove __MIN and __MAX columns from validation data
    val_columns = [col for col in val_df.columns if not col.endswith('__MIN') and not col.endswith('__MAX')]
    val_df = val_df[val_columns]
    
    # Dictionary to store results for validation (only need to compute once)
    model_results = defaultdict(dict)
    
    # Open output file
    with open('model_performance_analysis.txt', 'w') as f:
        def print_and_save(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)
        
        # Process validation data to find best indices (only need to do this once)
        for col in val_df.columns:
            if col == 'Step':
                continue
                
            base_model, variant = extract_model_info(col)
            if base_model is None or variant is None:
                continue
                
            # Convert to numeric, handling NaN values
            values = pd.to_numeric(val_df[col], errors='coerce')
            
            # Find the index of the maximum value (ignoring NaN)
            if not values.isna().all():
                max_idx = values.idxmin()
                max_value = values.loc[max_idx]
                
                # Calculate position among non-NaN values up to max_idx
                non_nan_values_up_to_max = values.loc[:max_idx].notna().sum()
                
                # Store the best validation performance and its index
                model_results[base_model][f'val_best_idx_{variant}'] = max_idx
                model_results[base_model][f'val_best_value_{variant}'] = max_value
                model_results[base_model][f'val_non_nan_position_{variant}'] = non_nan_values_up_to_max
        
        # Print validation indices and positions once at the beginning
        print_and_save("VALIDATION BEST INDICES AND POSITIONS (Used for all metrics)")
        print_and_save("=" * 80)
        print_and_save(f"{'Model':<25} {'Idx_1':<8} {'Idx_2':<8} {'Idx_3':<8} {'Pos_1':<8} {'Pos_2':<8} {'Pos_3':<8} {'Max_1':<10} {'Max_2':<10} {'Max_3':<10}")
        print_and_save("-" * 80)
        
        for model_type in sorted(model_results.keys()):
            indices = ['---', '---', '---']
            positions = ['---', '---', '---']
            max_values = ['---', '---', '---']
            
            for i, variant in enumerate(['1', '2', '3']):
                idx_key = f'val_best_idx_{variant}'
                pos_key = f'val_non_nan_position_{variant}'
                max_key = f'val_best_value_{variant}'
                
                if idx_key in model_results[model_type]:
                    indices[i] = str(model_results[model_type][idx_key])
                if pos_key in model_results[model_type]:
                    positions[i] = str(model_results[model_type][pos_key])
                if max_key in model_results[model_type]:
                    max_values[i] = f"{model_results[model_type][max_key]:.4f}"
            
            print_and_save(f"{model_type:<25} {indices[0]:<8} {indices[1]:<8} {indices[2]:<8} {positions[0]:<8} {positions[1]:<8} {positions[2]:<8} {max_values[0]:<10} {max_values[1]:<10} {max_values[2]:<10}")
        
        # Process each test file
        for test_file in sorted(test_files):
            metric_name = extract_metric_name(test_file)
            
            # Read test data
            test_df = pd.read_csv(test_file)
            
            # Remove __MIN and __MAX columns
            test_columns = [col for col in test_df.columns if not col.endswith('__MIN') and not col.endswith('__MAX')]
            test_df = test_df[test_columns]
            
            # Dictionary to store test results for this metric
            test_results = defaultdict(dict)
            
            # Process test data using the best indices from validation
            for col in test_df.columns:
                if col == 'Step':
                    continue
                    
                base_model, variant = extract_model_info(col)
                if base_model is None or variant is None:
                    continue
                    
                # Get the corresponding validation best index
                val_best_idx_key = f'val_best_idx_{variant}'
                if val_best_idx_key not in model_results[base_model]:
                    continue
                    
                best_idx = model_results[base_model][val_best_idx_key]
                
                # Convert to numeric, handling NaN values
                values = pd.to_numeric(test_df[col], errors='coerce')
                
                # Get test performance at the best validation index
                if best_idx < len(values) and not pd.isna(values.iloc[best_idx]):
                    test_value = values.iloc[best_idx]
                    test_results[base_model][f'test_value_{variant}'] = test_value
            
            # Create a summary table for this metric
            print_and_save(f"\nSummary Table - {metric_name.upper()}")
            print_and_save("=" * 80)
            print_and_save(f"{'Model':<25} {'Test Mean':<12} {'Test Var':<12} {'Test Std':<12} {'Val Mean':<12}")
            print_and_save("-" * 80)
            
            for model_type in sorted(test_results.keys()):
                test_values = []
                val_values = []
                
                for i, variant in enumerate(['1', '2', '3']):
                    test_key = f'test_value_{variant}'
                    val_key = f'val_best_value_{variant}'
                    
                    if test_key in test_results[model_type]:
                        test_values.append(test_results[model_type][test_key])
                        if val_key in model_results[model_type]:
                            val_values.append(model_results[model_type][val_key])
                
                if len(test_values) == 3:
                    mean_test = np.mean(test_values)
                    var_test = np.var(test_values, ddof=1)
                    std_test = np.std(test_values, ddof=1)
                    
                    val_mean_str = "N/A"
                    if len(val_values) == 3:
                        mean_val = np.mean(val_values)
                        val_mean_str = f"{mean_val:.4f}"
                    
                    print_and_save(f"{model_type:<25} {mean_test:<12.4f} {var_test:<12.6f} {std_test:<12.4f} {val_mean_str:<12}")
                else:
                    print_and_save(f"{model_type:<25} {'Incomplete data':<12} {'---':<12} {'---':<12} {'---':<12}")
        
        print_and_save(f"\nAnalysis saved to model_performance_analysis.txt")

if __name__ == "__main__":
    main()
