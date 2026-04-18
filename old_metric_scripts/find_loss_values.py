import pandas as pd
import numpy as np
import os
import glob

# Read the CSV file
df = pd.read_csv('wandb_export_3/wandb_loss_val.csv')

# Dictionary of model indices and positions from the balanced accuracy analysis
model_indices_bal_acc = {
    'biot_5sec_sl': {'idx': [323, 757, 8], 'pos': [32, 71, 3]},
    'biot_7sec_sl': {'idx': [138, 404, 48], 'pos': [17, 39, 9]},
    'biot_9sec_sl': {'idx': [1054, 761, 671], 'pos': [96, 71, 62]},
    'm+slstm_5sec_sl': {'idx': [125, 35, 789], 'pos': [16, 8, 72]},
    'm+slstm_7sec_sl': {'idx': [8, 1019, 0], 'pos': [3, 96, 1]},
    'm+slstm_9sec_sl': {'idx': [261, 276, 12], 'pos': [27, 28, 4]},
    'mlstm_5sec_sl': {'idx': [1062, 311, 982], 'pos': [97, 31, 91]},
    'mlstm_7sec_sl': {'idx': [40, 943, 203], 'pos': [8, 86, 22]},
    'mlstm_9sec_sl': {'idx': [138, 317, 392], 'pos': [17, 32, 38]},
    'slstm_5sec_sl': {'idx': [359, 534, 409], 'pos': [35, 50, 39]},
    'slstm_7sec_sl': {'idx': [629, 357, 509], 'pos': [58, 34, 49]},
    'slstm_9sec_sl': {'idx': [150, 982, 716], 'pos': [18, 91, 66]}
}

# Dictionary of model indices from the loss analysis
model_indices_loss = {
    'biot_5sec_sl': {'idx': [32, 6, 1], 'pos': [7, 3, 1]},
    'biot_7sec_sl': {'idx': [0, 0, 12], 'pos': [1, 1, 4]},
    'biot_9sec_sl': {'idx': [7, 0, 0], 'pos': [3, 1, 1]},
    'm+slstm_5sec_sl': {'idx': [0, 27, 0], 'pos': [1, 7, 1]},
    'm+slstm_7sec_sl': {'idx': [34, 780, 0], 'pos': [7, 74, 1]},
    'm+slstm_9sec_sl': {'idx': [226, 576, 12], 'pos': [24, 53, 4]},
    'mlstm_5sec_sl': {'idx': [79, 17, 30], 'pos': [12, 5, 7]},
    'mlstm_7sec_sl': {'idx': [7, 3, 0], 'pos': [3, 2, 1]},
    'mlstm_9sec_sl': {'idx': [79, 11, 202], 'pos': [12, 4, 22]},
    'slstm_5sec_sl': {'idx': [0, 0, 0], 'pos': [1, 1, 1]},
    'slstm_7sec_sl': {'idx': [40, 1, 46], 'pos': [8, 1, 9]},
    'slstm_9sec_sl': {'idx': [40, 6, 23], 'pos': [8, 3, 6]}
}

# Function to get loss value for a model at a specific index
def get_loss_value(model_name, index, variant):
    # Find the column that matches the model name and variant
    model_col = None
    for col in df.columns:
        if f"{model_name}_default_{variant}" in col and 'val/loss' in col and not ('MIN' in col or 'MAX' in col):
            model_col = col
            break
    
    if model_col is None:
        return None
    
    # Get the loss value at the specified index
    try:
        return df.iloc[index][model_col]
    except:
        return None

def find_matching_checkpoint(formatted_string):
    # Search in all wandb_checkpoints directories
    for checkpoint_dir in glob.glob('wandb_checkpoints/*'):
        # Search for files matching the pattern
        pattern = os.path.join(checkpoint_dir, f"{formatted_string}*.ckpt")
        matching_files = glob.glob(pattern)
        if matching_files:
            return matching_files[0]  # Return the first matching file
    return None

def print_results(model_indices, source_name):
    print(f"\nModel Loss Values at Indices ({source_name}):")
    print("=" * 50)
    
    # Dictionary to store formatted strings for each model
    formatted_strings = {}
    
    for model, data in model_indices.items():
        print(f"\n{model}:")
        model_strings = []
        for i in range(3):  # For each variant (1, 2, 3)
            idx = data['idx'][i]
            pos = data['pos'][i]
            loss = get_loss_value(model, idx, i+1)
            if loss is not None:
                print(f"  {model}_default_{i+1} (idx={idx}, pos={pos}): {loss}")
                # Add formatted string to list using position instead of index
                formatted_string = f"epoch={pos}-val_loss={loss:.4f}"
                model_strings.append(formatted_string)
                
                # Find matching checkpoint file
                checkpoint_file = find_matching_checkpoint(formatted_string)
                if checkpoint_file:
                    print(f"    Found checkpoint: {checkpoint_file}")
            else:
                print(f"  {model}_default_{i+1} (idx={idx}, pos={pos}): No data available")
                model_strings.append(f"epoch={pos}-val_loss=NA")
        
        # Store formatted strings for this model
        formatted_strings[model] = model_strings
    
    # Print formatted strings for each model
    print("\nFormatted Strings:")
    print("=" * 50)
    for model, strings in formatted_strings.items():
        print(f"\n{model}:")
        for i, string in enumerate(strings, 1):
            print(f"  Variant {i}: {string}")

# Toggle between the two sets of indices
use_loss_indices = False  # Set to False to use balanced accuracy indices

if use_loss_indices:
    print_results(model_indices_loss, "Loss Analysis")
else:
    print_results(model_indices_bal_acc, "Balanced Accuracy Analysis") 