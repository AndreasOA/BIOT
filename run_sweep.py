import itertools
import subprocess
import sys
from typing import Dict, List
import shutil  # For finding executables
import time  # Add at the top of the file with other imports
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from evaluate_model import load_model_from_checkpoint, evaluate_model, compute_metrics
from utils import TUEVLoader
import wandb
import glob
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

wandb.login(key=os.getenv("WANDB_API_KEY"))

class Args:
    """Class to hold arguments as attributes"""
    def __init__(self, params_dict):
        for key, value in params_dict.items():
            setattr(self, key, value)

def get_latest_wandb_run():
    """Get the run name from the output.log file and find the corresponding wandb folder"""
    try:
        # First get the run name from output.log
        run_name = None
        with open('wandb/latest-run/files/output.log', 'r') as f:
            for line in f:
                if "Full run name:" in line:
                    # Extract the run name after "Full run name: "
                    run_name = line.split("Full run name: ")[1].strip()
                    break
        
        if run_name is None:
            print("Warning: Could not find run name in output.log")
            return None, None

        # Now find the .wandb file and extract the run ID
        wandb_files = glob.glob('wandb/latest-run/run-*.wandb')
        if not wandb_files:
            print("Warning: Could not find .wandb file")
            return run_name, None

        # Get the first .wandb file (there should only be one)
        wandb_file = wandb_files[0]
        run_id = os.path.basename(wandb_file).replace('run-', '').replace('.wandb', '')

        # Find the corresponding folder in wandb directory
        # The folder name format is run-{date}-{time}-{run_id}
        wandb_folders = glob.glob(f'wandb/run-*_*-{run_id}')
        if not wandb_folders:
            print(f"Warning: Could not find folder for run ID {run_id}")
            return run_name, None

        return run_name, wandb_folders[0]

    except FileNotFoundError:
        print("Warning: output.log file not found")
        return None, None

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix as CSV table
    csv_save_path = save_path.replace('.png', '.csv')
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(csv_save_path, index=True, header=True)
    
    # Create and save the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_checkpoint(checkpoint_path, params, device):
    """Evaluate a single checkpoint and return metrics"""
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    
    # Convert params dictionary to Args object
    args = Args(params)
    
    # Load model
    lightning_model = load_model_from_checkpoint(checkpoint_path, args)
    print("Model loaded successfully!")
    
    # Prepare test data
    test_loader = prepare_test_dataloader(args)
    
    # Run evaluation
    predictions, labels = evaluate_model(lightning_model, test_loader, device)
    
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create save directory
    save_dir = os.path.dirname(checkpoint_path)
    
    # Compute metrics and save to file
    metrics = compute_metrics(predictions, labels, save_dir)
    
    # Create confusion matrix plot
    cm_save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, pred_classes, cm_save_path)
    
    # Save run summary for statistical analysis
    run_summary = {
        'model_type': f"BIOT_mlstm={params['mlstm']}_slstm={params['slstm']}",
        'seed': params['seed'],
        'metrics': metrics,
        'params': params
    }
    summary_path = os.path.join(save_dir, 'run_summary.json')
    with open(summary_path, 'w') as f:
        import json
        json.dump(run_summary, f, indent=2)
    
    return metrics

def generate_parameter_combinations() -> List[Dict]:
    # Define parameter grid based on sweep_config.yaml
    param_grid = {
        'lr': [0.001],
        'weight_decay': [1e-5],
        'batch_size': [128],
        'num_workers': [16],
        'sampling_rate': [250],
        'resampling_rate': [200],
        'token_size': [200],
        'hop_length': [100],
        'dataset': ['TUEV'],
        'model': ['BIOT'],
        'in_channels': [16],
        'n_classes': [6],
        'epochs': [100],
        'mlstm': [True],
        'slstm': [False],
        'dataset_size': [1.0],
        'val_ratio': [0.1],
        'secondsBeforeEvent': [4],
        'secondsAfterEvent': [4],
        'seed': [456, 999, 10]
    }
    #         'full_sample_method': ['attention', 'convolution'],
    # Get all keys and values
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        # Only keep combinations where secondsBeforeEvent equals secondsAfterEvent
        # AND ensure valid xLSTM configurations (at least one of mlstm or slstm must be True)
        if (param_dict['secondsBeforeEvent'] == param_dict['secondsAfterEvent'] and 
            (param_dict['mlstm'] or param_dict['slstm'])):  # At least one must be True
        #    if (param_dict['slstm'] == True and param_dict['secondsBeforeEvent'] == 3 and param_dict['secondsAfterEvent'] == 3) \
        #        or (param_dict['slstm'] == False and param_dict['secondsBeforeEvent'] == 2 and param_dict['secondsAfterEvent'] == 2):
            combinations.append(param_dict)
    

    # Print the number of combinations
    print(f"Total number of combinations: {len(combinations)}")
    print(f"Combinations: {combinations}")
    return combinations

def get_checkpoint_path(run_name, checkpoint_type):
    """Get the path to the checkpoint file in the specified folder"""
    checkpoint_dir = f"wandb_checkpoints/{run_name}/{checkpoint_type}"
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Get all files in the directory
    files = os.listdir(checkpoint_dir)
    if not files:
        return None
    
    # Return the path to the first (and should be only) file
    return os.path.join(checkpoint_dir, files[0])

def run_experiment(params: Dict, max_retries: int = 3, retry_delay: int = 30):
    # Try to find Python executable
    python_path = shutil.which('python3') or shutil.which('python') or sys.executable
    if not python_path:
        raise RuntimeError("Could not find Python executable")
    
    for attempt in range(max_retries):
        try:
            cmd = [python_path, 'run_multiclass_supervised.py']
            for key, value in params.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                cmd.extend([f'--{key}', str(value)])
            
            print(f"Running attempt {attempt + 1}/{max_retries}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                print("Output:", result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            # After training, get the latest wandb run
            run_name, wandb_folder = get_latest_wandb_run()
            if run_name is None:
                print("Warning: Could not find wandb run directory")
                return
            
            # Get checkpoint paths
            loss_checkpoint = get_checkpoint_path(run_name, "loss")
            acc_checkpoint = get_checkpoint_path(run_name, "bal_acc")
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Evaluate both checkpoints
            if loss_checkpoint and os.path.exists(loss_checkpoint):
                print("\nEvaluating best loss model...")
                evaluate_checkpoint(loss_checkpoint, params, device)
            else:
                print("\nNo loss checkpoint found")
            
            if acc_checkpoint and os.path.exists(acc_checkpoint):
                print("\nEvaluating best balanced accuracy model...")
                evaluate_checkpoint(acc_checkpoint, params, device)
            else:
                print("\nNo balanced accuracy checkpoint found")

            checkpoint_folder = f"wandb_checkpoints/{run_name}"
            
            # Create the target folder name based on parameters
            current_time = time.strftime("%Y%m%d_%H%M%S")
            target_folder_name = f"BIOT_mlstm={params['mlstm']}_slstm={params['slstm']}_secondsBeforeEvent={params['secondsBeforeEvent']}_secondsAfterEvent={params['secondsAfterEvent']}_seed={params['seed']}_{current_time}"
            target_folder = os.path.join("stored_runs", target_folder_name)
            
            # Create the stored_runs directory if it doesn't exist
            os.makedirs("stored_runs", exist_ok=True)
            
            # Move wandb folder if it exists
            if wandb_folder and os.path.exists(wandb_folder):
                target_wandb = os.path.join(target_folder, "wandb")
                os.makedirs(target_folder, exist_ok=True)
                print(f"\nMoving wandb folder to {target_wandb}")
                shutil.move(wandb_folder, target_wandb)
            
            # Move checkpoint folder if it exists
            if os.path.exists(checkpoint_folder):
                target_checkpoint = os.path.join(target_folder, "checkpoints")
                os.makedirs(target_folder, exist_ok=True)
                print(f"\nMoving checkpoint folder to {target_checkpoint}")
                shutil.move(checkpoint_folder, target_checkpoint)
            
            return  # Success
                    
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Moving to next experiment.")

def prepare_test_dataloader(args):
    """Prepare test dataloader"""
    root = f"datasets/{args.dataset}/edf"
    test_files = sorted(os.listdir(os.path.join(root, f"processed_eval_{args.secondsBeforeEvent}_{args.secondsAfterEvent}_{args.resampling_rate}")))
    
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, f"processed_eval_{args.secondsBeforeEvent}_{args.secondsAfterEvent}_{args.resampling_rate}"),
            test_files,
            args.resampling_rate,
            args.secondsBeforeEvent,
            args.secondsAfterEvent
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    
    print(f"Test set: {len(test_files)} files, {len(test_loader)} batches")
    return test_loader

def main():
    combinations = generate_parameter_combinations()
    print(f"Total number of experiments: {len(combinations)}")
    print("\nParameter combinations:")
    for i, params in enumerate(combinations, 1):
        print(f"\nCombination {i}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    for i, params in enumerate(combinations, 1):
        print(f"\nRunning experiment {i}/{len(combinations)}")
        run_experiment(params)

if __name__ == "__main__":
    main() 