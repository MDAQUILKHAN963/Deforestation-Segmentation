"""
model.py — U-Net with EfficientNet-B3 Encoder for Deforestation Segmentation.

Architecture:
    Encoder: EfficientNet-B3 (pretrained on ImageNet)
    Decoder: Custom upsampling path with skip connections
    Output:  Single-channel sigmoid (binary segmentation)

Built entirely with tf.keras — no external segmentation library required.
This is more transparent and impressive than using a pre-built wrapper.
"""

import tensorflow as tf
from tensorflow import keras


# ─── EfficientNet-B3 Skip Connection Layer Names ────────────────────────────────
# These layers produce feature maps at different spatial resolutions.
# We tap into them for the U-Net skip connections (encoder → decoder).

EFFICIENTNET_B3_SKIP_LAYERS = [
    "block2b_add",     # 1/4  resolution  — 48 filters
    "block3b_add",     # 1/8  resolution  — 48 filters
    "block4c_add",     # 1/16 resolution  — 96 filters
    "block6e_add",     # 1/32 resolution  — 232 filters
]


# ─── Decoder Block ──────────────────────────────────────────────────────────────

def decoder_block(x, skip, filters, block_name):
    """
    A single U-Net decoder block:
        1. Upsample (Conv2DTranspose)
        2. Concatenate with skip connection from encoder
        3. Two Conv2D layers with BatchNorm + ReLU

    Args:
        x: Input feature map from the previous decoder block
        skip: Skip connection from the encoder at matching resolution
        filters: Number of convolution filters
        block_name: Name prefix for Keras layers

    Returns:
        Output feature map at 2× spatial resolution
    """
    # Upsample
    x = keras.layers.Conv2DTranspose(
        filters, kernel_size=2, strides=2, padding="same",
        name=f"{block_name}_upsample"
    )(x)

    # Concatenate skip connection
    x = keras.layers.Concatenate(name=f"{block_name}_concat")([x, skip])

    # Conv block 1
    x = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        name=f"{block_name}_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = keras.layers.ReLU(name=f"{block_name}_relu1")(x)

    # Conv block 2
    x = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        name=f"{block_name}_conv2"
    )(x)
    x = keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = keras.layers.ReLU(name=f"{block_name}_relu2")(x)

    return x


# ─── Channel Adapter ────────────────────────────────────────────────────────────

def channel_adapter(inputs, name="channel_adapter"):
    """
    Learnable 1×1 convolution to project N input bands → 3 channels.
    This allows using ImageNet-pretrained EfficientNet even when
    input has N ≠ 3 bands (e.g., multi-spectral satellite imagery).

    Only applied when input channels ≠ 3.
    """
    x = keras.layers.Conv2D(
        3, kernel_size=1, padding="same", use_bias=True,
        name=f"{name}_conv1x1"
    )(inputs)
    x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    x = keras.layers.ReLU(name=f"{name}_relu")(x)
    return x


# ─── Build U-Net Model ──────────────────────────────────────────────────────────

def build_unet_efficientnet(
    input_shape: tuple = (256, 256, 3),
    classes: int = 1,
    encoder_weights: str = "imagenet",
    freeze_encoder: bool = False,
    decoder_filters: tuple = (256, 128, 64, 32),
    dropout_rate: float = 0.3,
):
    """
    Build a U-Net segmentation model with EfficientNet-B3 encoder.

    Architecture Overview:
        Input (256×256×N) → [Channel Adapter if N≠3] → EfficientNet-B3 Encoder
        → Bottleneck → Decoder (4 upsampling blocks with skip connections)
        → 1×1 Conv → Sigmoid → Output (256×256×1)

    Args:
        input_shape: Input image shape (H, W, C). C can be any number of bands.
        classes: Number of output classes (1 for binary segmentation).
        encoder_weights: Pretrained weights ('imagenet' or None).
        freeze_encoder: If True, freeze encoder weights (transfer learning).
        decoder_filters: Number of filters in each decoder block.
        dropout_rate: Spatial dropout rate before final output.

    Returns:
        tf.keras.Model: Compiled-ready U-Net model
    """
    # ── Input layer
    inputs = keras.layers.Input(shape=input_shape, name="input_image")

    # ── Channel adapter (if input is not 3-channel)
    n_channels = input_shape[-1]
    if n_channels != 3:
        x = channel_adapter(inputs, name="band_adapter")
        encoder_input_shape = (*input_shape[:2], 3)
    else:
        x = inputs
        encoder_input_shape = input_shape

    # ── Encoder: EfficientNet-B3
    backbone = keras.applications.EfficientNetB3(
        include_top=False,
        weights=encoder_weights,
        input_shape=encoder_input_shape,
    )

    if freeze_encoder:
        backbone.trainable = False

    # We need to run the input through the backbone manually
    # to extract intermediate skip connections
    encoder_output = backbone(x)

    # Extract skip connection feature maps
    skip_model = keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(name).output for name in EFFICIENTNET_B3_SKIP_LAYERS]
    )
    skip_outputs = skip_model(x)

    # ── Bottleneck (encoder output)
    bottleneck = encoder_output  # Deepest feature map

    # ── Decoder path
    # We go from deepest (smallest spatial) to shallowest (largest spatial)
    x = bottleneck
    for i, (filters, skip) in enumerate(
        zip(decoder_filters, reversed(skip_outputs))
    ):
        x = decoder_block(x, skip, filters, block_name=f"decoder_{i+1}")

    # ── Final upsample to match input resolution
    # EfficientNetB3 first downsamples by 2× at the stem, so we need one
    # more upsample after all decoder blocks
    x = keras.layers.Conv2DTranspose(
        32, kernel_size=2, strides=2, padding="same",
        name="final_upsample"
    )(x)
    x = keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False, name="final_conv1"
    )(x)
    x = keras.layers.BatchNormalization(name="final_bn1")(x)
    x = keras.layers.ReLU(name="final_relu1")(x)

    # ── Spatial dropout for regularization
    if dropout_rate > 0:
        x = keras.layers.SpatialDropout2D(dropout_rate, name="spatial_dropout")(x)

    # ── Output layer: 1×1 convolution with sigmoid activation
    outputs = keras.layers.Conv2D(
        classes, kernel_size=1, padding="same",
        activation="sigmoid", name="output_mask"
    )(x)

    # ── Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name="UNet_EfficientNetB3")

    return model


# ─── Model Summary Helper ───────────────────────────────────────────────────────

def print_model_summary(model):
    """Print a detailed model summary with parameter counts."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params

    print("\n" + "=" * 60)
    print(f"  Model: {model.name}")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print("=" * 60)
    print(f"  Total parameters:         {total_params:>12,}")
    print(f"  Trainable parameters:     {trainable_params:>12,}")
    print(f"  Non-trainable parameters: {non_trainable:>12,}")
    print("=" * 60 + "\n")


# ─── Quick Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Building U-Net with EfficientNet-B3 encoder...")

    # Test with 3-channel input (standard RGB)
    model_rgb = build_unet_efficientnet(input_shape=(256, 256, 3))
    print_model_summary(model_rgb)

    # Test with multi-band input (e.g., 6-band satellite imagery)
    model_multi = build_unet_efficientnet(input_shape=(256, 256, 6))
    print_model_summary(model_multi)

    # Test forward pass
    import numpy as np
    dummy_input = np.random.randn(2, 256, 256, 3).astype(np.float32)
    output = model_rgb.predict(dummy_input, verbose=0)
    print(f"[✓] Forward pass: input {dummy_input.shape} → output {output.shape}")
    print(f"[✓] Output range: [{output.min():.4f}, {output.max():.4f}]")
