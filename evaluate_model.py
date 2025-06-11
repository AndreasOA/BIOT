import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pyhealth.metrics import multiclass_metrics_fn
from sklearn.metrics import roc_auc_score, average_precision_score

from model import BIOTClassifier
from utils import TUEVLoader
from run_multiclass_supervised import LitModel_finetune


def load_model_from_checkpoint(checkpoint_path, args):
    """Load model from checkpoint file"""
    # Create the model architecture
    model = BIOTClassifier(
        n_classes=args.n_classes,
        n_channels=args.in_channels,
        n_fft=args.token_size,
        hop_length=args.hop_length,
        mlstm=args.mlstm,
        slstm=args.slstm,
    )
    
    # Load the lightning model from checkpoint
    lightning_model = LitModel_finetune.load_from_checkpoint(
        checkpoint_path, 
        args=args, 
        model=model
    )
    
    return lightning_model


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


def evaluate_model(lightning_model, test_loader, device):
    """Evaluate model on test data"""
    lightning_model.eval()
    lightning_model.to(device)
    
    all_predictions = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)
            
            # Get model predictions
            logits = lightning_model.model(X)
            
            all_predictions.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return predictions, labels


def compute_metrics(predictions, labels):
    """Compute comprehensive metrics"""
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Print confusion matrix-like statistics
    unique_classes = np.unique(labels)
    print("\nPer-class Statistics:")
    print("Class | Total | Correct | Accuracy")
    print("-" * 35)
    overall_correct = 0
    overall_total = 0
    for cls in unique_classes:
        mask = (labels == cls)
        total = np.sum(mask)
        correct = np.sum((pred_classes == labels) & mask)
        accuracy = correct / total if total > 0 else 0
        print(f"{int(cls):5d} | {total:5d} | {correct:7d} | {accuracy:8.2%}")
        overall_correct += correct
        overall_total += total
    print("-" * 35)
    print(f"Total | {overall_total:5d} | {overall_correct:7d} | {overall_correct/overall_total:8.2%}")
    print()
    
    # Calculate standard metrics
    try:
        metrics = multiclass_metrics_fn(
            labels, predictions, 
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        )
        
        print("Standard Metrics:")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Cohen Kappa:       {metrics['cohen_kappa']:.4f}")
        print(f"F1 Weighted:       {metrics['f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"Error computing standard metrics: {e}")
        metrics = {}
    
    # Compute probabilities for AUROC/AUC-PR
    try:
        probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()
        
        print("\nROC-AUC Metrics:")
        try:
            auroc_macro_ovr = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
            print(f"AUROC Macro (OvR):    {auroc_macro_ovr:.4f}")
        except Exception as e:
            print(f"Error in AUROC macro OvR: {e}")
            auroc_macro_ovr = 0.0
            
        try:
            auroc_weighted_ovr = roc_auc_score(labels, probs, average="weighted", multi_class="ovr")
            print(f"AUROC Weighted (OvR): {auroc_weighted_ovr:.4f}")
        except Exception as e:
            print(f"Error in AUROC weighted OvR: {e}")
            auroc_weighted_ovr = 0.0
            
        try:
            auroc_macro_ovo = roc_auc_score(labels, probs, average="macro", multi_class="ovo")
            print(f"AUROC Macro (OvO):    {auroc_macro_ovo:.4f}")
        except Exception as e:
            print(f"Error in AUROC macro OvO: {e}")
            auroc_macro_ovo = 0.0
            
        try:
            auroc_weighted_ovo = roc_auc_score(labels, probs, average="weighted", multi_class="ovo")
            print(f"AUROC Weighted (OvO): {auroc_weighted_ovo:.4f}")
        except Exception as e:
            print(f"Error in AUROC weighted OvO: {e}")
            auroc_weighted_ovo = 0.0
        
        print("\nAUC-PR Metrics:")
        try:
            aucpr_macro = average_precision_score(labels, probs, average="macro")
            print(f"AUC-PR Macro: {aucpr_macro:.4f}")
        except Exception as e:
            print(f"Error in AUC-PR macro: {e}")
            aucpr_macro = 0.0
            
        try:
            aucpr_micro = average_precision_score(labels, probs, average="micro")
            print(f"AUC-PR Micro: {aucpr_micro:.4f}")
        except Exception as e:
            print(f"Error in AUC-PR micro: {e}")
            aucpr_micro = 0.0
            
    except Exception as e:
        print(f"Error computing probability-based metrics: {e}")
    
    print("="*60)
    return metrics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model from checkpoint")
    
    # Model and checkpoint arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file")
    parser.add_argument("--model", type=str, default="BIOT", 
                        help="Model type")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="TUEV", 
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for data loading")
    
    # Model architecture arguments
    parser.add_argument("--in_channels", type=int, default=12,
                        help="Number of input channels")
    parser.add_argument("--n_classes", type=int, default=1,
                        help="Number of output classes")
    parser.add_argument("--token_size", type=int, default=200,
                        help="Token size")
    parser.add_argument("--hop_length", type=int, default=100,
                        help="Hop length")
    parser.add_argument("--mlstm", type=str2bool, default=True,
                        help="Use mLSTM")
    parser.add_argument("--slstm", type=str2bool, default=True,
                        help="Use sLSTM")
    
    # Data processing arguments
    parser.add_argument("--resampling_rate", type=int, default=200,
                        help="Resampling rate")
    parser.add_argument("--secondsBeforeEvent", type=int, default=2,
                        help="Seconds before event")
    parser.add_argument("--secondsAfterEvent", type=int, default=2,
                        help="Seconds after event")
    
    # Additional arguments that might be needed for model loading
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (for checkpoint compatibility)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (for checkpoint compatibility)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading model from: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    lightning_model = load_model_from_checkpoint(args.checkpoint_path, args)
    print("Model loaded successfully!")
    
    # Prepare test data
    test_loader = prepare_test_dataloader(args)
    
    # Run evaluation
    predictions, labels = evaluate_model(lightning_model, test_loader, device)
    
    # Compute and print metrics
    metrics = compute_metrics(predictions, labels)


if __name__ == "__main__":
    main() 