import itertools
import subprocess
import sys
from typing import Dict, List
import shutil  # For finding executables
import time  # Add at the top of the file with other imports

def generate_parameter_combinations() -> List[Dict]:
    # Define parameter grid based on sweep_config.yaml
    param_grid = {
        'lr': [0.0000005],
        'weight_decay': [1e-5],
        'batch_size': [128],
        'num_workers': [16],
        'sampling_rate': [200],
        'token_size': [200],
        'hop_length': [100],
        'dataset': ['TUEV'],
        'model': ['BIOT'],
        'in_channels': [16],
        'n_classes': [6],
        'epochs': [50],
        'sample_length': [10],
        'mlstm': [False, True],
        'slstm': [False, True],
        'dataset_size': [1.0],
        'val_ratio': [0.3],
        'use_full_sample': [False]
    }
    #         'full_sample_method': ['attention', 'convolution'],
    # Get all keys and values
    keys = param_grid.keys()
    values = param_grid.values()
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations

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
            
            if result.returncode == 0:
                return  # Success
            else:
                print(f"Command failed with return code {result.returncode}")
                if "Connection refused" in result.stderr:
                    print(f"Docker connection refused. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # If it's not a connection error, don't retry
                    break
                    
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Moving to next experiment.")

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