import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import multiclass_metrics_fn

import wandb

from model import (
    BIOTClassifier,
    LSTM
)
from utils import TUEVLoader, HARLoader


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # Log model parameters count
        self.save_hyperparameters(ignore=['model'])
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Log model statistics
        self.hparams.update({
            'total_parameters': n_params,
            'trainable_parameters': n_trainable_params,
            'non_trainable_parameters': n_params - n_trainable_params,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        })

    def training_step(self, batch, batch_idx):
        X, y = batch
        prod = self.model(X)
        train_loss = nn.CrossEntropyLoss()(prod, y)
        # Log with both formats
        self.log("train/loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/learning_rate", self.optimizers().param_groups[0]['lr'], on_step=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            val_loss = nn.CrossEntropyLoss()(convScore, y)
            # Log with both formats
            self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
            
            # Get predicted classes
            pred_classes = torch.argmax(convScore, dim=1)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
            # Store predictions along with other outputs
            self.validation_step_outputs.append((step_result, step_gt, pred_classes.cpu().numpy()))

    def on_validation_epoch_end(self):
        result = []
        gt = np.array([])
        predictions = np.array([])
        
        for out in self.validation_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])
            predictions = np.append(predictions, out[2])

        result = np.concatenate(result, axis=0)
        
        # Print confusion matrix-like statistics
        unique_classes = np.unique(gt)
        print("\nPrediction Statistics:")
        print("Class | Total | Correct | Accuracy")
        print("-" * 35)
        for cls in unique_classes:
            mask = (gt == cls)
            total = np.sum(mask)
            correct = np.sum((predictions == gt) & mask)
            accuracy = correct / total if total > 0 else 0
            print(f"{int(cls):5d} | {total:5d} | {correct:7d} | {accuracy:8.2%}")
        print("-" * 35)
        
        # Calculate metrics without the labels parameter
        result = multiclass_metrics_fn(
            gt, result, 
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        )
        
        # Handle potential NaN values in metrics
        for metric_name in ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]:
            if np.isnan(result[metric_name]):
                print(f"Warning: {metric_name} is NaN")
                result[metric_name] = 0.0
        
        self.log("val/acc", result["accuracy"], sync_dist=True)
        self.log("val/balanced_acc", result["balanced_accuracy"], sync_dist=True)
        self.log("val/cohen", result["cohen_kappa"], sync_dist=True)
        self.log("val/f1", result["f1_weighted"], sync_dist=True)
        print("\nOverall Metrics:")
        print(result)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            step_gt = y.cpu().numpy()
        # Store outputs instead of returning
        self.test_step_outputs.append((step_result, step_gt))

    def on_test_epoch_end(self):
        result = []
        gt = np.array([])
        for out in self.test_step_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])

        result = np.concatenate(result, axis=0)
        
        result = multiclass_metrics_fn(
            gt, result, 
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        )
        
        # Handle potential NaN values
        for metric_name in ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]:
            if np.isnan(result[metric_name]):
                print(f"Warning: {metric_name} is NaN")
                result[metric_name] = 0.0
        
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_balanced_acc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_cohen", result["cohen_kappa"], sync_dist=True)
        self.log("test_f1", result["f1_weighted"], sync_dist=True)
        
        self.test_step_outputs.clear()
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def get_train_val_split(train_files, train_sub, val_ratio=0.3, seed=42):
    """Generate consistent train/validation splits.
    
    Args:
        train_files: List of all training files
        train_sub: List of all training subjects
        val_ratio: Ratio of validation set size (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        val_files: List of validation files
        train_files: List of training files
    """
    rng = np.random.RandomState(seed)
    val_sub = rng.choice(train_sub, size=int(len(train_sub) * val_ratio), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]
    return val_files, train_files

def prepare_TUEV_dataloader(args):
    # Use a consistent seed across all random operations
    global_seed = 42  # Using same seed as train/val split
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    root = f"datasets/{args.dataset}/edf"

    train_files = sorted(os.listdir(os.path.join(root, "processed_train")))  # Sort for consistency
    train_sub = sorted(list(set([f.split("_")[0] for f in train_files])))  # Sort subjects
    print("train sub", len(train_sub))
    test_files = sorted(os.listdir(os.path.join(root, "processed_eval")))  # Sort for consistency

    # Apply dataset size reduction if specified
    if args.dataset_size < 1.0:
        rng = np.random.RandomState(global_seed)  # Use consistent seed
        n_train = int(len(train_sub) * args.dataset_size)
        train_sub = sorted(rng.choice(train_sub, size=n_train, replace=False))  # Sort after selection
        print(f"Reduced training set to {args.dataset_size*100}% ({n_train} subjects)")

    # Get consistent train/val split using same seed
    val_files, train_files = get_train_val_split(train_files, train_sub, val_ratio=args.val_ratio, seed=global_seed)

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    batch_size_printed = False

    for batch in train_loader:
        if not batch_size_printed:
            print(f"Training batch size: {batch[0].shape}")
            batch_size_printed = True

    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


class TestEpochEnd(pl.Callback):
    def __init__(self, test_dataloader):
        super().__init__()
        self.test_dataloader = test_dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        # Switch to eval mode
        pl_module.eval()
        
        # Run test evaluation
        test_outputs = []
        device = pl_module.device
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                convScore = pl_module.model(X)
                test_outputs.append((convScore.cpu().numpy(), y.cpu().numpy()))
        
        # Calculate metrics
        result = []
        gt = np.array([])
        for out in test_outputs:
            result.append(out[0])
            gt = np.append(gt, out[1])
        
        result = np.concatenate(result, axis=0)
        metrics = multiclass_metrics_fn(
            gt, result,
            metrics=["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        )
        
        # Log metrics
        pl_module.log("test/epoch_acc", metrics["accuracy"], sync_dist=True)
        pl_module.log("test/epoch_balanced_acc", metrics["balanced_accuracy"], sync_dist=True)
        pl_module.log("test/epoch_cohen", metrics["cohen_kappa"], sync_dist=True)
        pl_module.log("test/epoch_f1", metrics["f1_weighted"], sync_dist=True)
        
        # Switch back to training mode
        pl_module.train()

def supervised(args):
    # get data loaders
    if args.dataset == "TUEV":
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)

    else:
        raise NotImplementedError

    if args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
            mlstm=args.mlstm,
            slstm=args.slstm,
            use_full_sample=args.use_full_sample,
            full_sample_method=args.full_sample_method,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)

    # Replace logger setup with Wandb
    run_name = f"{args.dataset}-{args.model}"
    
    wandb_logger = WandbLogger(
        name=run_name,
        log_model=False,
        save_dir="./",
    )

    run_name = f"{wandb.run.name}-{run_name}"
    
    # Enhanced model logging
    wandb_logger.watch(
        model, 
        log="all",  # Log gradients and parameters
        log_freq=100,
        log_graph=True  # Log model graph
    )
    
    # Log model summary and architecture
    wandb_logger.experiment.config.update({
        "model_summary": str(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "architecture": {
            "type": args.model,
            "in_channels": args.in_channels,
            "n_classes": args.n_classes,
            "sampling_rate": args.sampling_rate
        }
    })

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"wandb_checkpoints/{run_name}",
        filename="{epoch:02d}-{val_loss:.4f}-{train_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        monitor="val/loss"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=20,
        verbose=False,
        mode="min"
    )

    # Create test callback
    test_callback = TestEpochEnd(test_loader)

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        enable_checkpointing=True,
        logger=wandb_logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, test_callback],  # Add test_callback
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # try:
    #     # Final test evaluation
    #     pretrain_result = trainer.test(
    #         model=lightning_model, ckpt_path="best", dataloaders=test_loader
    #     )[0]
    #     print(pretrain_result)
    # except Exception as e:
    #     print(f"Error during test evaluation: {e}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=16, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUEV", help="dataset")
    parser.add_argument(
        "--model", type=str, default="BIOT", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=12, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.3, help="validation ratio"
    )
    parser.add_argument(
        "--mlstm", type=str2bool, default=True, help="use mlstm"
    )
    parser.add_argument(
        "--slstm", type=str2bool, default=True, help="use slstm"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    parser.add_argument(
        "--dataset_size", type=float, default=1.0, 
        help="Fraction of dataset to use (0.0-1.0)"
    )
    parser.add_argument(
        "--use_full_sample", type=str2bool, default=False, 
        help="Process the full sample at once"
    )
    parser.add_argument(
        "--full_sample_method", type=str, default="attention", 
        choices=["attention", "convolution"],
        help="Method to process full sample: 'attention' or 'convolution'"
    )
    args = parser.parse_args()
    print(args)

    # Initialize wandb with all arguments as config
    wandb.init(
        project="biot-pretrain",
        config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "dataset": args.dataset,
            "val_ratio": args.val_ratio,
            "model": args.model,
            "in_channels": args.in_channels,
            "sample_length": args.sample_length,
            "mlstm": args.mlstm,
            "slstm": args.slstm,
            "n_classes": args.n_classes,
            "sampling_rate": args.sampling_rate,
            "token_size": args.token_size,
            "hop_length": args.hop_length,
            "pretrain_model_path": args.pretrain_model_path,
            "epochs": args.epochs,
            "dataset_size": args.dataset_size,
            "use_full_sample": args.use_full_sample,
            "full_sample_method": args.full_sample_method,
        },
        settings=wandb.Settings(start_method="fork")
    )

    supervised(args)
    
    # Finish the wandb run
    wandb.finish()
