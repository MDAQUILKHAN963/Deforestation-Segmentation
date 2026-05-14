"""
app.py — Gradio demo app for Deforestation Segmentation.

Provides an interactive web interface for:
    - Uploading satellite images
    - Running deforestation segmentation inference
    - Visualizing results with overlays and confidence maps

Usage:
    python app.py
    # Opens browser at http://localhost:7860
"""

import os
import numpy as np
import tensorflow as tf

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("[!] Gradio not installed. Install with: pip install gradio")

from src.model import build_unet_efficientnet
from src.losses import CombinedLoss
from src.metrics import IoUScore, FScore, PixelAccuracy
from src.dataset import IMG_SIZE, NUM_BANDS
from src.utils import get_rgb_from_multiband

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server


# ─── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH = "results/checkpoints/best_model.keras"
TITLE = "🌿 Deforestation Segmentation — AI Satellite Analysis"
DESCRIPTION = """
## 🛰️ Deep Learning for Deforestation Detection

Upload a satellite image to detect deforested regions using our **U-Net with EfficientNet-B3** model.

**Model Architecture:**
- Encoder: EfficientNet-B3 (ImageNet pretrained)
- Decoder: 4-level U-Net with skip connections
- Loss: Dice + Binary Focal Loss
- Input: 256×256 satellite imagery (RGB + NIR)

**Dataset:** MultiEarth 2023 (Sentinel-2 imagery from the Amazon rainforest)
"""


# ─── Global Model ────────────────────────────────────────────────────────────────

_model = None

def get_model():
    """Load model once and cache globally."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            custom_objects = {
                "CombinedLoss": CombinedLoss,
                "IoUScore": IoUScore,
                "FScore": FScore,
                "PixelAccuracy": PixelAccuracy,
            }
            _model = tf.keras.models.load_model(MODEL_PATH,
                                                 custom_objects=custom_objects)
            print(f"[✓] Loaded trained model from {MODEL_PATH}")
        else:
            print(f"[!] No trained model found at {MODEL_PATH}")
            print("[!] Using fresh (untrained) model for demo...")
            _model = build_unet_efficientnet(
                input_shape=(IMG_SIZE, IMG_SIZE, NUM_BANDS),
                encoder_weights="imagenet",
            )
    return _model


# ─── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an uploaded image for model inference.

    Handles:
        - Resizing to model input size
        - Channel adaptation (RGB→4 bands by duplicating green as NIR proxy)
        - Normalization to [0, 1]
    """
    # Ensure float32
    image = image.astype(np.float32)

    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0

    # Resize
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE]).numpy()

    # Adapt channels: if RGB (3 bands), add NIR proxy
    if image.shape[-1] == 3 and NUM_BANDS == 4:
        # Use green channel as NIR proxy (rough approximation)
        nir_proxy = image[..., 1:2] * 1.5  # Scale green band
        nir_proxy = np.clip(nir_proxy, 0, 1)
        image = np.concatenate([image, nir_proxy], axis=-1)
    elif image.shape[-1] > NUM_BANDS:
        image = image[..., :NUM_BANDS]

    return image


# ─── Inference ──────────────────────────────────────────────────────────────────

def segment_image(input_image):
    """
    Run deforestation segmentation on an uploaded image.

    Args:
        input_image: NumPy array from Gradio (H, W, 3) uint8

    Returns:
        Tuple of (overlay_image, heatmap_image, metrics_text)
    """
    if input_image is None:
        return None, None, "⚠️ Please upload an image."

    model = get_model()

    # Preprocess
    processed = preprocess_image(input_image)
    batch = np.expand_dims(processed, axis=0)  # Add batch dim

    # Predict
    prob_map = model.predict(batch, verbose=0)[0]  # Shape: (H, W, 1)
    binary_mask = (prob_map > 0.5).astype(np.float32)

    # ── Create overlay image
    display_img = input_image.astype(np.float32)
    if display_img.max() > 1.0:
        display_img = display_img / 255.0
    display_img = tf.image.resize(display_img, [IMG_SIZE, IMG_SIZE]).numpy()

    overlay = display_img.copy()
    mask_2d = tf.image.resize(binary_mask, [display_img.shape[0], display_img.shape[1]]).numpy().squeeze()

    # Red overlay for deforested areas
    overlay[mask_2d > 0.5, 0] = np.clip(overlay[mask_2d > 0.5, 0] + 0.4, 0, 1)
    overlay[mask_2d > 0.5, 1] = overlay[mask_2d > 0.5, 1] * 0.5
    overlay[mask_2d > 0.5, 2] = overlay[mask_2d > 0.5, 2] * 0.5

    # Green tint for forest areas
    overlay[mask_2d <= 0.5, 1] = np.clip(overlay[mask_2d <= 0.5, 1] + 0.15, 0, 1)

    overlay_uint8 = (overlay * 255).astype(np.uint8)

    # ── Create confidence heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    prob_2d = tf.image.resize(prob_map, [IMG_SIZE, IMG_SIZE]).numpy().squeeze()
    im = ax.imshow(prob_2d, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_title("Deforestation Probability Map", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Save to temporary file and load as array
    heatmap_path = "results/temp_heatmap.png"
    os.makedirs("results", exist_ok=True)
    fig.savefig(heatmap_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    heatmap_img = plt.imread(heatmap_path)
    heatmap_uint8 = (heatmap_img * 255).astype(np.uint8) if heatmap_img.max() <= 1.0 else heatmap_img.astype(np.uint8)

    # ── Compute metrics
    deforested_pct = float(mask_2d.mean()) * 100
    avg_confidence = float(prob_2d[mask_2d > 0.5].mean()) if (mask_2d > 0.5).any() else 0
    total_pixels = mask_2d.size
    deforested_pixels = int((mask_2d > 0.5).sum())

    metrics_text = f"""
### 📊 Analysis Results

| Metric | Value |
|--------|-------|
| **Deforestation Area** | {deforested_pct:.1f}% of image |
| **Deforested Pixels** | {deforested_pixels:,} / {total_pixels:,} |
| **Avg. Confidence** | {avg_confidence:.1%} |
| **Prediction Threshold** | 0.50 |

#### 🔍 Interpretation
{"🔴 **Significant deforestation detected!** The model identifies substantial forest loss in this image." if deforested_pct > 20 else "🟡 **Moderate deforestation detected.** Some areas show signs of forest degradation." if deforested_pct > 5 else "🟢 **Minimal deforestation detected.** The forest appears largely intact."}
"""

    return overlay_uint8, heatmap_uint8, metrics_text


# ─── Build Gradio Interface ─────────────────────────────────────────────────────

def create_app():
    """Create and return the Gradio app interface."""
    if not HAS_GRADIO:
        raise ImportError("Gradio is required. Install with: pip install gradio")

    with gr.Blocks(
        title="Deforestation Segmentation",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="red",
        ),
    ) as app:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📸 Upload Satellite Image",
                    type="numpy",
                    height=300,
                )
                analyze_btn = gr.Button(
                    "🔍 Analyze Deforestation",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_overlay = gr.Image(
                    label="🗺️ Segmentation Overlay",
                    type="numpy",
                    height=300,
                )

        with gr.Row():
            with gr.Column(scale=1):
                output_heatmap = gr.Image(
                    label="🌡️ Confidence Heatmap",
                    type="numpy",
                    height=300,
                )
            with gr.Column(scale=1):
                output_metrics = gr.Markdown(
                    label="📊 Metrics",
                    value="Upload an image and click **Analyze** to see results.",
                )

        # Event handler
        analyze_btn.click(
            fn=segment_image,
            inputs=[input_image],
            outputs=[output_overlay, output_heatmap, output_metrics],
        )

        gr.Markdown("""
        ---
        **🏗️ Built with:** TensorFlow 2.x | Keras | EfficientNet-B3 | U-Net Architecture

        **📄 Reference:** arXiv:2307.04916 — Deep Multimodal Learning on Satellite Imagery

        **📂 Dataset:** MultiEarth 2023 — Sentinel-2 imagery from the Amazon rainforest
        """)

    return app


# ─── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
