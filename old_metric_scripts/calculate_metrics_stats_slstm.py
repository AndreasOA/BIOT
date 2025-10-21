import numpy as np
import pandas as pd

# Extract metrics from the three runs
runs_data = {
    'seed_456': {
        'accuracy': 0.7564,
        'balanced_accuracy': 0.5332,
        'cohen_kappa': 0.5480,
        'f1_weighted': 0.7644,
        'auroc_macro_ovr': 0.8801,
        'auroc_weighted_ovr': 0.9103,
        'auroc_macro_ovo': 0.8372,
        'auroc_weighted_ovo': 0.8742,
        'aucpr_macro': 0.5203,
        'aucpr_micro': 0.8185
    },
    'seed_123': {
        'accuracy': 0.7635,
        'balanced_accuracy': 0.5474,
        'cohen_kappa': 0.5673,
        'f1_weighted': 0.7707,
        'auroc_macro_ovr': 0.8998,
        'auroc_weighted_ovr': 0.9155,
        'auroc_macro_ovo': 0.8518,
        'auroc_weighted_ovo': 0.8881,
        'aucpr_macro': 0.5433,
        'aucpr_micro': 0.8359
    },
    'seed_42': {
        'accuracy': 0.7318,
        'balanced_accuracy': 0.5815,
        'cohen_kappa': 0.5344,
        'f1_weighted': 0.7559,
        'auroc_macro_ovr': 0.8660,
        'auroc_weighted_ovr': 0.9020,
        'auroc_macro_ovo': 0.8327,
        'auroc_weighted_ovo': 0.8711,
        'aucpr_macro': 0.5337,
        'aucpr_micro': 0.7917
    }
}

# Convert to DataFrame for easier analysis
df = pd.DataFrame(runs_data).T

# Calculate mean and std for each metric
results = {}
for metric in df.columns:
    values = df[metric].values
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Sample standard deviation
    results[metric] = {'mean': mean_val, 'std': std_val}

# Print results in a formatted table
print("=" * 60)
print("METRIC STATISTICS ACROSS THREE RUNS")
print("=" * 60)
print(f"{'Metric':<25} {'Mean':<10} {'Std':<10}")
print("-" * 60)

for metric, stats in results.items():
    print(f"{metric:<25} {stats['mean']:<10.4f} {stats['std']:<10.4f}")

print("=" * 60)

# Also print individual run values for reference
print("\nINDIVIDUAL RUN VALUES:")
print("=" * 60)
for metric in df.columns:
    print(f"\n{metric}:")
    for run, value in df[metric].items():
        print(f"  {run}: {value:.4f}") 