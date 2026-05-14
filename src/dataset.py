"""
dataset.py — tf.data pipeline for satellite imagery segmentation.

Handles:
    - Loading multi-band GeoTIFF files (Sentinel-2) via rasterio
    - Band stacking and normalization
    - Augmentation via albumentations
    - tf.data.Dataset creation with batching and prefetching
    - Synthetic data generation for testing without real data
"""

import os
import glob
import numpy as np
import tensorflow as tf

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[!] rasterio not installed — GeoTIFF loading disabled. Using synthetic data.")

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[!] albumentations not installed — augmentation disabled.")


# ─── Configuration ──────────────────────────────────────────────────────────────

# Default image size for the model
IMG_SIZE = 256

# Number of bands to use (Sentinel-2: B2=Blue, B3=Green, B4=Red, B8=NIR)
# Using 4 bands: RGB + NIR for vegetation analysis (NDVI-capable)
NUM_BANDS = 4

# Band indices in Sentinel-2 L2A data (0-indexed)
# B2 (Blue)=1, B3 (Green)=2, B4 (Red)=3, B8 (NIR)=7
SENTINEL2_BAND_INDICES = [1, 2, 3, 7]


# ─── GeoTIFF I/O ────────────────────────────────────────────────────────────────

def load_geotiff(filepath: str) -> np.ndarray:
    """
    Load a GeoTIFF file using rasterio.

    Args:
        filepath: Path to the GeoTIFF file

    Returns:
        numpy array of shape (H, W, C) with float32 values
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to load GeoTIFF files")

    with rasterio.open(filepath) as src:
        # Read all bands: shape (C, H, W) → transpose to (H, W, C)
        image = src.read().astype(np.float32)
        image = np.transpose(image, (1, 2, 0))

    return image


def load_mask(filepath: str) -> np.ndarray:
    """
    Load a binary segmentation mask from a GeoTIFF.

    Args:
        filepath: Path to the mask GeoTIFF file

    Returns:
        numpy array of shape (H, W, 1) with values {0, 1}
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required to load GeoTIFF files")

    with rasterio.open(filepath) as src:
        mask = src.read(1).astype(np.float32)

    # Ensure binary values
    mask = (mask > 0).astype(np.float32)

    return mask[..., np.newaxis]  # Shape: (H, W, 1)


def select_bands(image: np.ndarray, band_indices: list = None) -> np.ndarray:
    """
    Select specific bands from a multi-band image.

    Args:
        image: Multi-band image (H, W, C)
        band_indices: List of band indices to select. If None, use default.

    Returns:
        Image with only selected bands (H, W, len(band_indices))
    """
    if band_indices is None:
        band_indices = SENTINEL2_BAND_INDICES

    # Clamp indices to available bands
    max_bands = image.shape[-1]
    valid_indices = [i for i in band_indices if i < max_bands]

    if len(valid_indices) < len(band_indices):
        print(f"[!] Warning: requested {len(band_indices)} bands, "
              f"only {len(valid_indices)} available. Using first {min(max_bands, len(band_indices))} bands.")
        valid_indices = list(range(min(max_bands, len(band_indices))))

    return image[..., valid_indices]


# ─── Normalization ──────────────────────────────────────────────────────────────

def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize satellite image bands.

    Args:
        image: Input image (H, W, C)
        method: 'minmax' for [0,1] scaling, 'zscore' for zero-mean unit-variance

    Returns:
        Normalized image as float32
    """
    image = image.astype(np.float32)

    if method == "minmax":
        # Per-band min-max normalization
        for c in range(image.shape[-1]):
            band = image[..., c]
            bmin, bmax = band.min(), band.max()
            if bmax - bmin > 0:
                image[..., c] = (band - bmin) / (bmax - bmin)
            else:
                image[..., c] = 0.0

    elif method == "zscore":
        # Per-band z-score normalization
        for c in range(image.shape[-1]):
            band = image[..., c]
            mean, std = band.mean(), band.std()
            if std > 0:
                image[..., c] = (band - mean) / std
            else:
                image[..., c] = 0.0

    return image


# ─── Augmentation ────────────────────────────────────────────────────────────────

def get_augmentation_pipeline(is_train: bool = True):
    """
    Create an albumentations augmentation pipeline.

    Training augmentations include spatial and color transforms.
    Validation/test uses only resize.

    Args:
        is_train: Whether this is for training (True) or eval (False)

    Returns:
        albumentations.Compose pipeline
    """
    if not HAS_ALBUMENTATIONS:
        return None

    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
                border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                fill_value=0, p=0.3
            ),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
        ])


def apply_augmentation(image: np.ndarray, mask: np.ndarray,
                       transform) -> tuple:
    """
    Apply albumentations augmentation to image-mask pair.

    Args:
        image: Input image (H, W, C)
        mask: Binary mask (H, W, 1)
        transform: albumentations Compose pipeline

    Returns:
        Tuple of (augmented_image, augmented_mask)
    """
    if transform is None:
        return image, mask

    mask_2d = mask.squeeze(-1) if mask.ndim == 3 else mask

    augmented = transform(image=image, mask=mask_2d)

    aug_image = augmented["image"].astype(np.float32)
    aug_mask = augmented["mask"].astype(np.float32)

    if aug_mask.ndim == 2:
        aug_mask = aug_mask[..., np.newaxis]

    return aug_image, aug_mask


# ─── Dataset Discovery ──────────────────────────────────────────────────────────

def discover_file_pairs(data_dir: str, image_pattern: str = "*.tif",
                        mask_suffix: str = "_mask") -> list:
    """
    Discover image-mask file pairs in a directory.

    Expects naming convention where mask files have the same name as
    image files but with a suffix (e.g., image001.tif → image001_mask.tif).

    Args:
        data_dir: Root data directory to search
        image_pattern: Glob pattern for image files
        mask_suffix: Suffix that distinguishes mask files from image files

    Returns:
        List of (image_path, mask_path) tuples
    """
    all_files = glob.glob(os.path.join(data_dir, "**", image_pattern), recursive=True)

    # Separate image files from mask files
    mask_files = {f for f in all_files if mask_suffix in os.path.basename(f)}
    image_files = [f for f in all_files if f not in mask_files]

    pairs = []
    for img_path in sorted(image_files):
        base, ext = os.path.splitext(img_path)
        mask_path = f"{base}{mask_suffix}{ext}"

        if mask_path in mask_files:
            pairs.append((img_path, mask_path))

    print(f"[✓] Found {len(pairs)} image-mask pairs in {data_dir}")
    return pairs


# ─── tf.data Pipeline ───────────────────────────────────────────────────────────

def create_tf_dataset(
    file_pairs: list,
    batch_size: int = 8,
    is_train: bool = True,
    num_bands: int = NUM_BANDS,
    img_size: int = IMG_SIZE,
    buffer_size: int = 256,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from a list of (image_path, mask_path) pairs.

    Pipeline:
        1. Generator yields (image, mask) numpy arrays
        2. .map() applies normalization + augmentation
        3. .batch() + .prefetch() for performance

    Args:
        file_pairs: List of (image_path, mask_path) tuples
        batch_size: Batch size
        is_train: Whether to apply training augmentations
        num_bands: Number of bands in the output images
        img_size: Target spatial resolution
        buffer_size: Shuffle buffer size (only when is_train=True)

    Returns:
        tf.data.Dataset yielding (image, mask) batches
    """
    transform = get_augmentation_pipeline(is_train)

    def data_generator():
        """Python generator that loads and preprocesses image-mask pairs."""
        indices = list(range(len(file_pairs)))
        if is_train:
            np.random.shuffle(indices)

        for idx in indices:
            img_path, mask_path = file_pairs[idx]

            try:
                # Load GeoTIFF files
                image = load_geotiff(img_path)
                mask = load_mask(mask_path)

                # Select bands
                image = select_bands(image)

                # Normalize
                image = normalize_image(image, method="minmax")

                # Resize if needed (before augmentation for consistency)
                if image.shape[0] != img_size or image.shape[1] != img_size:
                    image = tf.image.resize(image, [img_size, img_size]).numpy()
                    mask = tf.image.resize(mask, [img_size, img_size],
                                           method="nearest").numpy()

                # Augmentation
                image, mask = apply_augmentation(image, mask, transform)

                yield image, mask

            except Exception as e:
                print(f"[!] Error loading {img_path}: {e}")
                continue

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, num_bands), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
        ),
    )

    # Shuffle, batch, prefetch
    if is_train:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ─── Synthetic Data Generator ────────────────────────────────────────────────────

def create_synthetic_dataset(
    num_samples: int = 200,
    img_size: int = IMG_SIZE,
    num_bands: int = NUM_BANDS,
    batch_size: int = 8,
    is_train: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create a synthetic dataset for testing the pipeline without real data.

    Generates random satellite-like images with circular/rectangular
    'deforestation' patches as masks.

    Args:
        num_samples: Number of synthetic samples to generate
        img_size: Image spatial resolution
        num_bands: Number of image bands
        batch_size: Batch size
        is_train: Whether this is for training
        seed: Random seed

    Returns:
        tf.data.Dataset yielding (image, mask) batches
    """
    rng = np.random.RandomState(seed)
    transform = get_augmentation_pipeline(is_train)

    def synthetic_generator():
        for i in range(num_samples):
            # Create a forest-like background (green-ish in RGB bands)
            image = np.zeros((img_size, img_size, num_bands), dtype=np.float32)

            # Band 0 (Blue): low values for forest
            image[..., 0] = rng.uniform(0.05, 0.15, (img_size, img_size))
            # Band 1 (Green): moderate-high for forest
            image[..., 1] = rng.uniform(0.2, 0.5, (img_size, img_size))
            # Band 2 (Red): low for forest
            image[..., 2] = rng.uniform(0.05, 0.2, (img_size, img_size))

            # Extra bands (NIR, etc.): moderate values
            for b in range(3, num_bands):
                image[..., b] = rng.uniform(0.3, 0.7, (img_size, img_size))

            # Create deforestation mask with random patches
            mask = np.zeros((img_size, img_size), dtype=np.float32)

            num_patches = rng.randint(1, 5)
            for _ in range(num_patches):
                # Random circular or rectangular patch
                cx = rng.randint(20, img_size - 20)
                cy = rng.randint(20, img_size - 20)
                radius = rng.randint(10, 50)

                if rng.random() > 0.5:
                    # Circular patch
                    yy, xx = np.ogrid[:img_size, :img_size]
                    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
                    mask[circle] = 1.0
                else:
                    # Rectangular patch
                    h = rng.randint(15, 60)
                    w = rng.randint(15, 60)
                    y1, y2 = max(0, cy - h // 2), min(img_size, cy + h // 2)
                    x1, x2 = max(0, cx - w // 2), min(img_size, cx + w // 2)
                    mask[y1:y2, x1:x2] = 1.0

            # Make deforested areas look different (brown-ish)
            deforested = mask > 0.5
            image[deforested, 0] = rng.uniform(0.3, 0.5, deforested.sum())  # More blue
            image[deforested, 1] = rng.uniform(0.2, 0.35, deforested.sum())  # Less green
            image[deforested, 2] = rng.uniform(0.3, 0.5, deforested.sum())  # More red
            for b in range(3, num_bands):
                image[deforested, b] = rng.uniform(0.1, 0.3, deforested.sum())

            # Add noise
            noise = rng.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

            mask = mask[..., np.newaxis]  # Shape: (H, W, 1)

            # Apply augmentation
            if transform is not None:
                image, mask = apply_augmentation(image, mask, transform)

            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        synthetic_generator,
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, num_bands), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
        ),
    )

    if is_train:
        dataset = dataset.shuffle(min(num_samples, 256))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ─── Split Helpers ───────────────────────────────────────────────────────────────

def train_val_test_split(file_pairs: list, val_ratio: float = 0.15,
                         test_ratio: float = 0.15, seed: int = 42) -> tuple:
    """
    Split file pairs into train / validation / test sets.

    Args:
        file_pairs: List of (image_path, mask_path) tuples
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(file_pairs))
    rng.shuffle(indices)

    n_test = int(len(file_pairs) * test_ratio)
    n_val = int(len(file_pairs) * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_pairs = [file_pairs[i] for i in train_idx]
    val_pairs = [file_pairs[i] for i in val_idx]
    test_pairs = [file_pairs[i] for i in test_idx]

    print(f"[✓] Split: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    return train_pairs, val_pairs, test_pairs


# ─── Quick Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Testing synthetic dataset pipeline...")

    ds = create_synthetic_dataset(num_samples=32, batch_size=8)

    for batch_images, batch_masks in ds.take(1):
        print(f"[✓] Batch images shape: {batch_images.shape}")
        print(f"[✓] Batch masks shape:  {batch_masks.shape}")
        print(f"[✓] Image value range:  [{batch_images.numpy().min():.3f}, {batch_images.numpy().max():.3f}]")
        print(f"[✓] Mask unique values: {np.unique(batch_masks.numpy())}")
        print(f"[✓] Mask positive ratio: {batch_masks.numpy().mean():.3f}")

    print("[✓] Dataset pipeline test passed!")
