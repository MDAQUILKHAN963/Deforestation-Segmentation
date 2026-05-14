"""
train.py — Training script for Deforestation Segmentation.

Trains a U-Net with EfficientNet-B3 encoder using model.fit() API.
Optimized for CPU-only training with appropriate batch sizes and epochs.

Usage:
    python -m src.train                           # Train with synthetic data
    python -m src.train --data_dir data/multiearth # Train with real data
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

# Project imports
from src.model import build_unet_efficientnet, print_model_summary
from src.losses import CombinedLoss, DiceLoss, BinaryFocalLoss
from src.metrics import IoUScore, FScore, PixelAccuracy
from src.dataset import (
    create_tf_dataset,
    create_synthetic_dataset,
    discover_file_pairs,
    train_val_test_split,
    IMG_SIZE,
    NUM_BANDS,
)
from src.utils import set_seed, plot_training_history


# ─── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "img_size": IMG_SIZE,
    "num_bands": NUM_BANDS,
    "batch_size": 4,          # Small for CPU training
    "num_synthetic_train": 200,
    "num_synthetic_val": 50,

    # Model
    "encoder_weights": "imagenet",
    "freeze_encoder": False,
    "dropout_rate": 0.3,

    # Training
    "epochs": 15,             # Reasonable for CPU
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,

    # Loss
    "dice_weight": 1.0,
    "focal_weight": 1.0,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,

    # Paths
    "checkpoint_dir": "results/checkpoints",
    "log_dir": "results/logs",
    "history_plot_path": "results/training_history.png",
}


# ─── Learning Rate Schedule ─────────────────────────────────────────────────────

def create_lr_schedule(initial_lr: float, total_steps: int) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Create a Cosine Decay learning rate schedule.

    Starts at initial_lr and smoothly decays to near-zero,
    then optionally restarts. Cosine annealing helps escape
    local minima and improves final convergence.

    Args:
        initial_lr: Starting learning rate
        total_steps: Total number of training steps (epochs × steps_per_epoch)

    Returns:
        CosineDecay schedule
    """
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=1e-6,  # Minimum learning rate
    )


# ─── Callbacks ──────────────────────────────────────────────────────────────────

def create_callbacks(config: dict) -> list:
    """
    Create training callbacks.

    Callbacks:
        1. ModelCheckpoint — save best model by validation IoU
        2. EarlyStopping — stop if val_loss doesn't improve
        3. CSVLogger — log metrics to CSV for later analysis
        4. ReduceLROnPlateau — reduce LR if val_loss plateaus

    Args:
        config: Training configuration dictionary

    Returns:
        List of Keras callbacks
    """
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        # Save best model based on validation IoU
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["checkpoint_dir"], "best_model.keras"),
            monitor="val_iou_score",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        # Save latest model at every epoch
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config["checkpoint_dir"], "latest_model.keras"),
            save_best_only=False,
            verbose=0,
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),

        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(config["log_dir"], f"training_log_{timestamp}.csv"),
            separator=",",
            append=False,
        ),

        # Reduce LR on plateau (backup if cosine decay isn't enough)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    return callbacks


# ─── Training Function ──────────────────────────────────────────────────────────

def train(config: dict, data_dir: str = None):
    """
    Main training function.

    Args:
        config: Training configuration dictionary
        data_dir: Path to real dataset directory (None → use synthetic data)
    """
    print("\n" + "=" * 60)
    print("  DEFORESTATION SEGMENTATION — TRAINING")
    print("=" * 60)

    # ── Seed for reproducibility
    set_seed(42)

    # ── Prepare datasets
    if data_dir and os.path.exists(data_dir):
        print(f"\n[1/5] Loading real dataset from: {data_dir}")
        file_pairs = discover_file_pairs(data_dir)

        if len(file_pairs) == 0:
            print("[!] No file pairs found. Falling back to synthetic data.")
            train_ds = create_synthetic_dataset(
                num_samples=config["num_synthetic_train"],
                batch_size=config["batch_size"], is_train=True
            )
            val_ds = create_synthetic_dataset(
                num_samples=config["num_synthetic_val"],
                batch_size=config["batch_size"], is_train=False, seed=99
            )
        else:
            train_pairs, val_pairs, _ = train_val_test_split(file_pairs)
            train_ds = create_tf_dataset(
                train_pairs, batch_size=config["batch_size"], is_train=True
            )
            val_ds = create_tf_dataset(
                val_pairs, batch_size=config["batch_size"], is_train=False
            )
    else:
        print(f"\n[1/5] Creating synthetic dataset ({config['num_synthetic_train']} train, "
              f"{config['num_synthetic_val']} val)")
        train_ds = create_synthetic_dataset(
            num_samples=config["num_synthetic_train"],
            batch_size=config["batch_size"],
            is_train=True,
        )
        val_ds = create_synthetic_dataset(
            num_samples=config["num_synthetic_val"],
            batch_size=config["batch_size"],
            is_train=False,
            seed=99,
        )

    # Verify data shapes
    for images, masks in train_ds.take(1):
        print(f"  Train batch — images: {images.shape}, masks: {masks.shape}")
    for images, masks in val_ds.take(1):
        print(f"  Val batch   — images: {images.shape}, masks: {masks.shape}")

    # ── Build model
    print(f"\n[2/5] Building U-Net with EfficientNet-B3 encoder...")
    input_shape = (config["img_size"], config["img_size"], config["num_bands"])

    model = build_unet_efficientnet(
        input_shape=input_shape,
        classes=1,
        encoder_weights=config["encoder_weights"],
        freeze_encoder=config["freeze_encoder"],
        dropout_rate=config["dropout_rate"],
    )
    print_model_summary(model)

    # ── Compile model
    print(f"\n[3/5] Compiling model...")

    # Loss
    loss_fn = CombinedLoss(
        dice_weight=config["dice_weight"],
        focal_weight=config["focal_weight"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"],
    )

    # Optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Metrics
    metrics = [
        IoUScore(name="iou_score"),
        FScore(name="f_score"),
        PixelAccuracy(name="pixel_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    print("  Loss:      CombinedLoss (Dice + Focal)")
    print(f"  Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"  Metrics:   IoU, F1, Pixel Accuracy")

    # ── Create callbacks
    callbacks = create_callbacks(config)
    print(f"\n[4/5] Callbacks: {len(callbacks)} active")

    # ── Train
    print(f"\n[5/5] Starting training for {config['epochs']} epochs...")
    print("-" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save results
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    # Plot training history
    try:
        os.makedirs(os.path.dirname(config["history_plot_path"]), exist_ok=True)
        plot_training_history(history, save_path=config["history_plot_path"])
    except Exception as e:
        print(f"[!] Could not save training plot: {e}")

    # Save config
    config_path = os.path.join(config["log_dir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[✓] Config saved to {config_path}")

    # Print best metrics
    best_epoch = np.argmax(history.history.get("val_iou_score", [0]))
    print(f"\n  Best Epoch: {best_epoch + 1}")
    for key in ["val_loss", "val_iou_score", "val_f_score", "val_pixel_accuracy"]:
        if key in history.history:
            val = history.history[key][best_epoch]
            print(f"  {key}: {val:.4f}")

    print(f"\n[✓] Best model saved to: {config['checkpoint_dir']}/best_model.keras")

    return model, history


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-Net for Deforestation Segmentation"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real dataset directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Batch size (keep small for CPU)")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Initial learning rate")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze EfficientNet encoder weights")
    parser.add_argument("--num_synthetic_train", type=int,
                        default=DEFAULT_CONFIG["num_synthetic_train"],
                        help="Number of synthetic training samples")
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG["img_size"],
                        help="Image size (height and width)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "freeze_encoder": args.freeze_encoder,
        "num_synthetic_train": args.num_synthetic_train,
        "img_size": args.img_size,
    })

    model, history = train(config, data_dir=args.data_dir)
