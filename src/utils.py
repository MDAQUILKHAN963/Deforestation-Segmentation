"""
utils.py — Utility functions for the Deforestation Segmentation project.

Includes:
    - Reproducibility helpers (seed setting)
    - Visualization helpers (predictions, training curves)
    - Band extraction for display
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ─── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[✓] Random seed set to {seed}")


# ─── Band Extraction ────────────────────────────────────────────────────────────

def get_rgb_from_multiband(image: np.ndarray, rgb_indices: tuple = (2, 1, 0)) -> np.ndarray:
    """
    Extract RGB bands from a multi-band satellite image for visualization.

    Args:
        image: Multi-band image array of shape (H, W, C)
        rgb_indices: Tuple of (R, G, B) band indices. Default assumes
                     Sentinel-2 band ordering where B4=Red, B3=Green, B2=Blue.

    Returns:
        RGB image normalized to [0, 1] for matplotlib display.
    """
    if image.ndim == 2:
        # Single band — return as grayscale
        img = image.copy()
    elif image.shape[-1] >= 3:
        r, g, b = rgb_indices
        img = np.stack([image[..., r], image[..., g], image[..., b]], axis=-1)
    else:
        # Fewer than 3 bands — just use first band as grayscale
        img = image[..., 0]

    # Normalize to [0, 1] using percentile stretch for better contrast
    p2, p98 = np.percentile(img, (2, 98))
    if p98 - p2 > 0:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    else:
        img = np.clip(img, 0, 1)

    return img.astype(np.float32)


# ─── Prediction Visualization ───────────────────────────────────────────────────

def plot_prediction(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str = None,
    title: str = None,
    figsize: tuple = (15, 5),
):
    """
    Plot a 3-panel visualization: Input RGB | Ground Truth Mask | Predicted Mask.

    Args:
        image: Input satellite image (H, W, C) — will extract RGB automatically
        gt_mask: Ground truth binary mask (H, W) or (H, W, 1)
        pred_mask: Predicted binary mask (H, W) or (H, W, 1)
        save_path: Optional path to save the figure
        title: Optional super title
        figsize: Figure size
    """
    # Prepare images
    rgb = get_rgb_from_multiband(image)
    gt = gt_mask.squeeze()
    pred = pred_mask.squeeze()

    # Custom colormap for masks: green = forest (0), red = deforested (1)
    mask_cmap = mcolors.ListedColormap(["#2d6a4f", "#e63946"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: RGB input
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite Image (RGB)", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Ground truth
    axes[1].imshow(gt, cmap=mask_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground Truth Mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(pred, cmap=mask_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Predicted Mask", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved prediction plot to {save_path}")

    plt.show()
    plt.close(fig)


def plot_prediction_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str = None,
    alpha: float = 0.4,
    figsize: tuple = (15, 5),
):
    """
    Plot prediction with colored overlay on the satellite image.

    Args:
        image: Input satellite image (H, W, C)
        gt_mask: Ground truth binary mask
        pred_mask: Predicted binary mask
        save_path: Optional save path
        alpha: Overlay transparency
        figsize: Figure size
    """
    rgb = get_rgb_from_multiband(image)
    gt = gt_mask.squeeze()
    pred = pred_mask.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Original
    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: GT overlay
    axes[1].imshow(rgb)
    overlay_gt = np.zeros((*gt.shape, 4))
    overlay_gt[gt > 0.5] = [1, 0, 0, alpha]       # Red for deforested
    overlay_gt[gt <= 0.5] = [0, 0.7, 0.3, alpha * 0.3]  # Green for forest
    axes[1].imshow(overlay_gt)
    axes[1].set_title("Ground Truth Overlay", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Prediction overlay
    axes[2].imshow(rgb)
    overlay_pred = np.zeros((*pred.shape, 4))
    overlay_pred[pred > 0.5] = [1, 0, 0, alpha]
    overlay_pred[pred <= 0.5] = [0, 0.7, 0.3, alpha * 0.3]
    axes[2].imshow(overlay_pred)
    axes[2].set_title("Prediction Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved overlay plot to {save_path}")

    plt.show()
    plt.close(fig)


# ─── Training History Visualization ─────────────────────────────────────────────

def plot_training_history(history, save_path: str = None, figsize: tuple = (16, 10)):
    """
    Plot training/validation loss and metric curves from model.fit() history.

    Args:
        history: tf.keras.callbacks.History object or dict of lists
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    # Determine which metrics are available
    metric_keys = [k for k in hist.keys() if not k.startswith("val_") and k != "lr"]
    n_metrics = len(metric_keys)

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2, 2, figsize=figsize, squeeze=False
    )
    axes = axes.flatten()

    colors = sns.color_palette("husl", n_metrics)

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        epochs = range(1, len(hist[key]) + 1)

        # Training curve
        ax.plot(epochs, hist[key], "-o", color=colors[i],
                label=f"Train {key}", markersize=4, linewidth=2)

        # Validation curve (if exists)
        val_key = f"val_{key}"
        if val_key in hist:
            ax.plot(epochs, hist[val_key], "--s", color=colors[i],
                    label=f"Val {key}", markersize=4, linewidth=2, alpha=0.7)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(key.replace("_", " ").title(), fontsize=11)
        ax.set_title(key.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Training History", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved training history to {save_path}")

    plt.show()
    plt.close(fig)


# ─── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: str = None, figsize: tuple = (8, 6)):
    """
    Plot a confusion matrix for binary segmentation results.

    Args:
        y_true: Flattened ground truth labels
        y_pred: Flattened predicted labels
        save_path: Optional save path
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true.flatten(), y_pred.flatten())

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["Forest", "Deforested"],
        yticklabels=["Forest", "Deforested"],
        ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Saved confusion matrix to {save_path}")

    plt.show()
    plt.close(fig)


# ─── Summary Printer ────────────────────────────────────────────────────────────

def print_metrics_summary(metrics_dict: dict):
    """Pretty-print a dictionary of evaluation metrics."""
    print("\n" + "=" * 50)
    print("       EVALUATION METRICS SUMMARY")
    print("=" * 50)
    for name, value in metrics_dict.items():
        print(f"  {name:<25s} : {value:.4f}")
    print("=" * 50 + "\n")
