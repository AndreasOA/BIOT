# Model Evaluation Scripts

This directory contains scripts to evaluate trained models from wandb checkpoints.

## Files

- `evaluate_model.py` - Main evaluation script that loads a model and computes test metrics
- `list_checkpoints.py` - Helper script to list available checkpoints
- `EVALUATION_README.md` - This file

## Quick Start

### 1. List Available Checkpoints

First, see what checkpoints are available:

```bash
python list_checkpoints.py
```

This will show all checkpoint files in the `wandb_checkpoints/` directory with their details and provide example usage commands.

### 2. Evaluate a Model

Use the `evaluate_model.py` script to evaluate a specific checkpoint:

```bash
python evaluate_model.py \
    --checkpoint_path "wandb_checkpoints/run-name/epoch=10-val_loss=0.1234.ckpt" \
    --dataset TUEV \
    --batch_size 512 \
    --in_channels 12 \
    --n_classes 1 \
    --resampling_rate 200 \
    --secondsBeforeEvent 2 \
    --secondsAfterEvent 2
```

## Arguments for evaluate_model.py

### Required Arguments
- `--checkpoint_path`: Path to the checkpoint file to evaluate

### Model Configuration (must match training configuration)
- `--model`: Model type (default: "BIOT")
- `--in_channels`: Number of input channels (default: 12)
- `--n_classes`: Number of output classes (default: 1)
- `--token_size`: Token size (default: 200)
- `--hop_length`: Hop length (default: 100)
- `--mlstm`: Use mLSTM (default: True)
- `--slstm`: Use sLSTM (default: True)

### Data Configuration (must match training configuration)
- `--dataset`: Dataset name (default: "TUEV")
- `--resampling_rate`: Resampling rate (default: 200)
- `--secondsBeforeEvent`: Seconds before event (default: 2)
- `--secondsAfterEvent`: Seconds after event (default: 2)

### Evaluation Configuration
- `--batch_size`: Batch size for evaluation (default: 512)
- `--num_workers`: Number of data loading workers (default: 16)

## Output

The evaluation script will output:

1. **Per-class Statistics**: Shows total samples, correct predictions, and accuracy for each class
2. **Standard Metrics**:
   - Accuracy
   - Balanced Accuracy
   - Cohen Kappa
   - F1 Weighted
3. **ROC-AUC Metrics**:
   - Macro and Weighted (One-vs-Rest and One-vs-One)
4. **AUC-PR Metrics**:
   - Macro and Micro

## Important Notes

- **Configuration Matching**: The model architecture and data processing parameters must match those used during training
- **Data Availability**: The test data must be available in the expected directory structure
- **GPU Usage**: The script will automatically use GPU if available
- **Error Handling**: The script includes robust error handling for metric computation

## Example Output

```
============================================================
TEST RESULTS
============================================================

Per-class Statistics:
Class | Total | Correct | Accuracy
-----------------------------------
    0 |  1234 |    1100 |   89.14%
    1 |   567 |     523 |   92.24%
-----------------------------------
Total |  1801 |    1623 |   90.12%

Standard Metrics:
Accuracy:          0.9012
Balanced Accuracy: 0.9069
Cohen Kappa:       0.8024
F1 Weighted:       0.9015

ROC-AUC Metrics:
AUROC Macro (OvR):    0.9456
AUROC Weighted (OvR): 0.9523
AUROC Macro (OvO):    0.9456
AUROC Weighted (OvO): 0.9523

AUC-PR Metrics:
AUC-PR Macro: 0.9234
AUC-PR Micro: 0.9345
============================================================
```

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Verify the checkpoint path is correct
2. **Data directory not found**: Ensure the processed test data exists
3. **Configuration mismatch**: Check that model parameters match training configuration
4. **Memory issues**: Reduce batch size if you encounter out-of-memory errors

### Getting Help

Run with `--help` to see all available options:

```bash
python evaluate_model.py --help
``` 