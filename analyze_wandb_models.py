#!/usr/bin/env python3
"""
Analyze WandB export CSV files to find the highest validation accuracy for each model
across all export files, group models by base name, and calculate means.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from collections import defaultdict

def extract_model_name(column_name):
    """Extract model name from column name, removing the metric suffix."""
    # Remove the "- val/balanced_acc" part and any __MIN, __MAX suffixes
    model_name = column_name.replace(" - val/balanced_acc", "")
    model_name = re.sub(r"__(?:MIN|MAX)$", "", model_name)
    return model_name.strip()

def extract_base_model_name(model_name):
    """Extract base model name without the _1, _2, _3 suffix."""
    # Remove _1, _2, _3 suffixes at the end
    base_name = re.sub(r"_[123]$", "", model_name)
    return base_name

def main():
    # Read the main CSV file from wandb_export_3
    main_csv_path = "wandb_export_3/wandb_bal_acc_val.csv"
    print(f"Reading main CSV file: {main_csv_path}")
    
    main_df = pd.read_csv(main_csv_path)
    
    # Get all value columns (excluding Step and __MIN, __MAX columns)
    value_columns = [col for col in main_df.columns 
                     if col != "Step" and not col.endswith("__MIN") and not col.endswith("__MAX")]
    
    print(f"\nFound {len(value_columns)} model columns")
    
    # For each model, find the row with the highest value
    model_max_values = {}
    model_max_rows = {}
    
    for col in value_columns:
        model_name = extract_model_name(col)
        
        # Convert to numeric, replacing empty strings and non-numeric values with NaN
        numeric_values = pd.to_numeric(main_df[col], errors='coerce')
        
        if not numeric_values.isna().all():  # If there are any valid values
            max_idx = numeric_values.idxmax()
            max_value = numeric_values.iloc[max_idx]
            step = main_df.iloc[max_idx]['Step']
            
            model_max_values[model_name] = max_value
            model_max_rows[model_name] = {
                'step': step,
                'value': max_value,
                'row_index': max_idx
            }
    
    print(f"\nFound maximum values for {len(model_max_values)} models")
    
    # Read all other CSV files in wandb_export directory
    export_files = glob.glob("wandb_export/*.csv")
    print(f"\nFound {len(export_files)} files in wandb_export directory")
    
    # For each model, collect values from all export files at the best step
    model_all_values = defaultdict(list)
    
    for export_file in export_files:
        print(f"Processing {export_file}")
        try:
            df = pd.read_csv(export_file)
            
            for model_name, max_info in model_max_rows.items():
                best_step = max_info['step']
                
                # Find the corresponding column name in this file
                possible_cols = [col for col in df.columns if extract_model_name(col) == model_name]
                
                if possible_cols:
                    col = possible_cols[0]  # Take the first match (should be the main value column)
                    
                    # Find the row with the best step
                    step_rows = df[df['Step'] == best_step]
                    if not step_rows.empty:
                        value = pd.to_numeric(step_rows[col].iloc[0], errors='coerce')
                        if not pd.isna(value):
                            model_all_values[model_name].append(value)
        
        except Exception as e:
            print(f"Error processing {export_file}: {e}")
    
    # Print results for each model
    print("\n" + "="*80)
    print("RESULTS FOR EACH MODEL")
    print("="*80)
    
    for model_name in sorted(model_max_values.keys()):
        max_info = model_max_rows[model_name]
        all_vals = model_all_values[model_name]
        
        print(f"\nModel: {model_name}")
        print(f"  Best step: {max_info['step']}")
        print(f"  Best value: {max_info['value']:.6f}")
        print(f"  Values from all files: {[f'{v:.6f}' for v in all_vals]}")
        if all_vals:
            print(f"  Mean across files: {np.mean(all_vals):.6f}")
    
    # Group models by base name and calculate means
    print("\n" + "="*80)
    print("GROUPED MODELS (removing _1, _2, _3 suffixes)")
    print("="*80)
    
    base_model_groups = defaultdict(list)
    base_model_all_values = defaultdict(list)
    
    for model_name, max_value in model_max_values.items():
        base_name = extract_base_model_name(model_name)
        base_model_groups[base_name].append((model_name, max_value))
        
        # Also collect all values for this base model
        all_vals = model_all_values[model_name]
        base_model_all_values[base_name].extend(all_vals)
    
    for base_name in sorted(base_model_groups.keys()):
        models = base_model_groups[base_name]
        all_vals = base_model_all_values[base_name]
        
        print(f"\nBase Model: {base_name}")
        for model_name, max_value in models:
            print(f"  {model_name}: {max_value:.6f}")
        
        if len(models) > 1:
            mean_max = np.mean([max_val for _, max_val in models])
            print(f"  Mean of max values: {mean_max:.6f}")
        
        if all_vals:
            print(f"  Mean across all files: {np.mean(all_vals):.6f}")
    
    # Print the actual rows with highest val values
    print("\n" + "="*80)
    print("ROWS WITH HIGHEST VAL VALUES")
    print("="*80)
    
    for model_name in sorted(model_max_values.keys()):
        max_info = model_max_rows[model_name]
        row_idx = max_info['row_index']
        step = max_info['step']
        value = max_info['value']
        
        print(f"\nModel: {model_name}")
        print(f"  Step: {step}, Value: {value:.6f}")
        print(f"  Row index: {row_idx}")
        
        # Find the column name in the original dataframe
        matching_cols = [col for col in main_df.columns if extract_model_name(col) == model_name]
        if matching_cols:
            col_name = matching_cols[0]
            row_data = main_df.iloc[row_idx]
            print(f"  Full row data for step {step}:")
            print(f"    {col_name}: {row_data[col_name]}")

if __name__ == "__main__":
    main() 