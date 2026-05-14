"""
predict.py — Inference and visualization for Deforestation Segmentation.

Loads a trained model and generates predictions on test data.
Produces side-by-side visualizations and computes evaluation metrics.

Usage:
    python -m src.predict                                  # Predict on synthetic test data
    python -m src.predict --model results/checkpoints/best_model.keras  # Custom model path
    python -m src.predict --data_dir data/multiearth       # Predict on real data
"""

import os
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.model import build_unet_efficientnet
from src.losses import CombinedLoss
from src.metrics import (
    IoUScore, FScore, PixelAccuracy,
    compute_iou, compute_f1, compute_pixel_accuracy,
)
from src.dataset import (
    create_tf_dataset,
    create_synthetic_dataset,
    discover_file_pairs,
    train_val_test_split,
    IMG_SIZE,
    NUM_BANDS,
)
from src.utils import (
    plot_prediction,
    plot_prediction_overlay,
    plot_confusion_matrix,
    print_metrics_summary,
    get_rgb_from_multiband,
)


# ─── Load Model ─────────────────────────────────────────────────────────────────

def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model from disk.

    Handles custom objects (losses, metrics) registration.

    Args:
        model_path: Path to the saved .keras or .h5 model file

    Returns:
        Loaded tf.keras.Model
    """
    custom_objects = {
        "CombinedLoss": CombinedLoss,
        "IoUScore": IoUScore,
        "FScore": FScore,
        "PixelAccuracy": PixelAccuracy,
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"[✓] Model loaded from: {model_path}")
    print(f"    Input shape:  {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")

    return model


# ─── Predict on a Batch ─────────────────────────────────────────────────────────

def predict_batch(model: tf.keras.Model, images: np.ndarray,
                  threshold: float = 0.5) -> tuple:
    """
    Run inference on a batch of images.

    Args:
        model: Trained model
        images: Input images, shape (B, H, W, C)
        threshold: Binarization threshold for probability maps

    Returns:
        Tuple of (probability_maps, binary_masks)
    """
    prob_maps = model.predict(images, verbose=0)
    binary_masks = (prob_maps > threshold).astype(np.float32)

    return prob_maps, binary_masks


# ─── Evaluate on Full Test Set ───────────────────────────────────────────────────

def evaluate_test_set(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """
    Evaluate model on the full test dataset.

    Args:
        model: Trained model
        test_ds: Test tf.data.Dataset

    Returns:
        Dictionary of evaluation metrics
    """
    all_iou = []
    all_f1 = []
    all_acc = []

    all_y_true = []
    all_y_pred = []

    print("[*] Evaluating on test set...")

    for batch_idx, (images, masks) in enumerate(test_ds):
        prob_maps, pred_masks = predict_batch(model, images.numpy())

        # Compute per-batch metrics
        iou = compute_iou(masks.numpy(), prob_maps)
        f1 = compute_f1(masks.numpy(), prob_maps)
        acc = compute_pixel_accuracy(masks.numpy(), prob_maps)

        all_iou.append(iou)
        all_f1.append(f1)
        all_acc.append(acc)

        # Collect for confusion matrix
        all_y_true.append(masks.numpy().flatten())
        all_y_pred.append(pred_masks.flatten())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}: IoU={iou:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # Aggregate metrics
    metrics = {
        "IoU (Jaccard)": np.mean(all_iou),
        "F1 Score (Dice)": np.mean(all_f1),
        "Pixel Accuracy": np.mean(all_acc),
        "IoU Std": np.std(all_iou),
        "F1 Std": np.std(all_f1),
    }

    # Confusion matrix data
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    return metrics, y_true_all, y_pred_all


# ─── Visualize Predictions ──────────────────────────────────────────────────────

def visualize_predictions(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    num_samples: int = 8,
    save_dir: str = "results/predictions",
):
    """
    Generate and save prediction visualizations.

    Creates side-by-side plots of Input | Ground Truth | Prediction
    for a selection of test samples.

    Args:
        model: Trained model
        test_ds: Test dataset
        num_samples: Number of samples to visualize
        save_dir: Directory to save prediction images
    """
    os.makedirs(save_dir, exist_ok=True)

    sample_count = 0

    for batch_images, batch_masks in test_ds:
        prob_maps, pred_masks = predict_batch(model, batch_images.numpy())

        for i in range(batch_images.shape[0]):
            if sample_count >= num_samples:
                return

            image = batch_images[i].numpy()
            gt_mask = batch_masks[i].numpy()
            pred_mask = pred_masks[i]
            prob_map = prob_maps[i]

            # Side-by-side plot
            save_path = os.path.join(save_dir, f"prediction_{sample_count:04d}.png")
            plot_prediction(
                image, gt_mask, pred_mask,
                save_path=save_path,
                title=f"Sample {sample_count + 1}"
            )

            # Overlay plot
            overlay_path = os.path.join(save_dir, f"overlay_{sample_count:04d}.png")
            plot_prediction_overlay(
                image, gt_mask, pred_mask,
                save_path=overlay_path
            )

            # Confidence heatmap
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(prob_map.squeeze(), cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title(f"Confidence Heatmap — Sample {sample_count + 1}",
                         fontsize=13, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Deforestation Probability")
            plt.tight_layout()
            heatmap_path = os.path.join(save_dir, f"heatmap_{sample_count:04d}.png")
            fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            sample_count += 1

    print(f"[✓] Saved {sample_count} prediction visualizations to {save_dir}")


# ─── Main Prediction Pipeline ───────────────────────────────────────────────────

def predict(
    model_path: str = "results/checkpoints/best_model.keras",
    data_dir: str = None,
    num_samples: int = 50,
    batch_size: int = 4,
    num_visualize: int = 8,
    save_dir: str = "results/predictions",
):
    """
    Full prediction pipeline: load model → predict → evaluate → visualize.

    Args:
        model_path: Path to trained model
        data_dir: Path to real data (None → synthetic)
        num_samples: Number of test samples (for synthetic data)
        batch_size: Batch size for inference
        num_visualize: Number of samples to create visualizations for
        save_dir: Directory to save results
    """
    print("\n" + "=" * 60)
    print("  DEFORESTATION SEGMENTATION — INFERENCE")
    print("=" * 60)

    # ── Load model
    print(f"\n[1/4] Loading model...")
    if os.path.exists(model_path):
        model = load_trained_model(model_path)
    else:
        print(f"[!] Model not found at {model_path}")
        print("[!] Building a fresh (untrained) model for testing...")
        model = build_unet_efficientnet(
            input_shape=(IMG_SIZE, IMG_SIZE, NUM_BANDS),
            encoder_weights="imagenet",
        )

    # ── Prepare test dataset
    print(f"\n[2/4] Preparing test dataset...")
    if data_dir and os.path.exists(data_dir):
        file_pairs = discover_file_pairs(data_dir)
        if len(file_pairs) > 0:
            _, _, test_pairs = train_val_test_split(file_pairs)
            test_ds = create_tf_dataset(
                test_pairs, batch_size=batch_size, is_train=False
            )
        else:
            print("[!] No file pairs found. Using synthetic data.")
            test_ds = create_synthetic_dataset(
                num_samples=num_samples, batch_size=batch_size,
                is_train=False, seed=123
            )
    else:
        print(f"  Using synthetic test data ({num_samples} samples)")
        test_ds = create_synthetic_dataset(
            num_samples=num_samples, batch_size=batch_size,
            is_train=False, seed=123
        )

    # ── Evaluate
    print(f"\n[3/4] Running evaluation...")
    metrics, y_true, y_pred = evaluate_test_set(model, test_ds)
    print_metrics_summary(metrics)

    # ── Visualize
    print(f"\n[4/4] Generating visualizations...")
    # Need fresh dataset for visualization (generators are one-shot)
    if data_dir and os.path.exists(data_dir):
        file_pairs = discover_file_pairs(data_dir)
        if len(file_pairs) > 0:
            _, _, test_pairs = train_val_test_split(file_pairs)
            viz_ds = create_tf_dataset(
                test_pairs, batch_size=batch_size, is_train=False
            )
        else:
            viz_ds = create_synthetic_dataset(
                num_samples=num_samples, batch_size=batch_size,
                is_train=False, seed=123
            )
    else:
        viz_ds = create_synthetic_dataset(
            num_samples=num_samples, batch_size=batch_size,
            is_train=False, seed=123
        )

    visualize_predictions(model, viz_ds, num_samples=num_visualize, save_dir=save_dir)

    # Confusion matrix
    try:
        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    except Exception as e:
        print(f"[!] Could not generate confusion matrix: {e}")

    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Results saved to: {save_dir}")

    return metrics


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Deforestation Segmentation model"
    )
    parser.add_argument("--model", type=str,
                        default="results/checkpoints/best_model.keras",
                        help="Path to trained model file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to real test data directory")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of synthetic test samples")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--num_visualize", type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument("--save_dir", type=str,
                        default="results/predictions",
                        help="Directory to save prediction results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        model_path=args.model,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_visualize=args.num_visualize,
        save_dir=args.save_dir,
    )
