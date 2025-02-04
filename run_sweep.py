import itertools
import subprocess
import sys
from typing import Dict, List
import shutil  # For finding executables

def generate_parameter_combinations() -> List[Dict]:
    # Define parameter grid based on sweep_config.yaml
    param_grid = {
        'lr': [0.001, 0.0001, 0.00001],
        'weight_decay': [1e-5],
        'batch_size': [128, 256],
        'num_workers': [16],
        'sampling_rate': [100],
        'token_size': [200],
        'hop_length': [100],
        'dataset': ['TUEV'],
        'model': ['BIOT'],
        'in_channels': [16],
        'n_classes': [6],
        'epochs': [50],
        'sample_length': [5],
        'mlstm': [False, True],
        'slstm': [False, True],
        'dataset_size': [0.6, 1.0],
    }
    
    # Get all keys and values
    keys = param_grid.keys()
    values = param_grid.values()
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations

def run_experiment(params: Dict):
    # Try to find Python executable
    python_path = shutil.which('python3') or shutil.which('python') or sys.executable
    if not python_path:
        raise RuntimeError("Could not find Python executable")
    
    cmd = [python_path, 'run_multiclass_supervised.py']
    for key, value in params.items():
        # Convert boolean values to lowercase strings for argparse
        if isinstance(value, bool):
            value = str(value).lower()
        cmd.extend([f'--{key}', str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output for debugging
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")

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