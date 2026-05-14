"""
download_data.py — Dataset download helper for MultiEarth 2023.

The MultiEarth 2023 dataset contains Sentinel-1, Sentinel-2, and Landsat-8
satellite imagery with binary deforestation masks from the Amazon rainforest.

Download source: Azure Blob Storage (requires azcopy or manual download)
Reference: https://sites.google.com/view/rainforest-challenge/multiearth-2023

Usage:
    python -m src.download_data                    # Download to data/multiearth
    python -m src.download_data --output_dir data  # Custom output directory
    python -m src.download_data --generate_synthetic --num_samples 500  # Generate synthetic data
"""

import os
import sys
import subprocess
import shutil
import argparse

import numpy as np


# ─── Constants ──────────────────────────────────────────────────────────────────

AZURE_BLOB_URL = "https://rainforestchallenge.blob.core.windows.net/multiearth2023-dataset-final/"
GITHUB_REPO = "https://github.com/MIT-AI-Accelerator/multiearth-challenge.git"
DEFAULT_OUTPUT = os.path.join("data", "multiearth")


# ─── Check for azcopy ───────────────────────────────────────────────────────────

def check_azcopy() -> bool:
    """Check if azcopy is available on the system."""
    return shutil.which("azcopy") is not None


# ─── Download with azcopy ────────────────────────────────────────────────────────

def download_with_azcopy(output_dir: str) -> bool:
    """
    Download MultiEarth 2023 dataset using azcopy.

    Args:
        output_dir: Local directory to save the dataset

    Returns:
        True if download was successful
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Downloading MultiEarth 2023 dataset to: {output_dir}")
    print(f"[*] Source: {AZURE_BLOB_URL}")
    print(f"[!] This may take a while (dataset is several GB)...\n")

    cmd = [
        "azcopy", "cp",
        AZURE_BLOB_URL,
        output_dir,
        "--recursive"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n[✓] Download complete! Data saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[✗] Download failed with error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("[✗] azcopy not found. Please install it first.")
        return False


# ─── Synthetic Data Generation ───────────────────────────────────────────────────

def generate_synthetic_geotiffs(output_dir: str, num_samples: int = 200,
                                img_size: int = 256, num_bands: int = 4,
                                seed: int = 42):
    """
    Generate synthetic satellite imagery GeoTIFF files for testing.

    Creates image-mask pairs that mimic the structure of real satellite data
    with synthetic deforestation patterns.

    Args:
        output_dir: Directory to save generated files
        num_samples: Number of image-mask pairs to generate
        img_size: Image spatial resolution
        num_bands: Number of spectral bands
        seed: Random seed
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        HAS_RASTERIO = True
    except ImportError:
        HAS_RASTERIO = False

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    print(f"[*] Generating {num_samples} synthetic satellite image-mask pairs...")
    print(f"    Output: {output_dir}")
    print(f"    Size: {img_size}×{img_size}, Bands: {num_bands}")

    for i in range(num_samples):
        # ── Generate image
        image = np.zeros((num_bands, img_size, img_size), dtype=np.float32)

        # Create forest-like spectral signatures
        image[0] = rng.uniform(0.02, 0.08, (img_size, img_size))   # Blue
        image[1] = rng.uniform(0.03, 0.12, (img_size, img_size))   # Green
        image[2] = rng.uniform(0.02, 0.07, (img_size, img_size))   # Red
        if num_bands > 3:
            image[3] = rng.uniform(0.15, 0.45, (img_size, img_size))  # NIR

        # ── Generate deforestation mask
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        num_patches = rng.randint(0, 6)
        for _ in range(num_patches):
            cx = rng.randint(20, img_size - 20)
            cy = rng.randint(20, img_size - 20)

            shape_type = rng.choice(["circle", "rect", "irregular"])

            if shape_type == "circle":
                radius = rng.randint(8, 45)
                yy, xx = np.ogrid[:img_size, :img_size]
                circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
                mask[circle] = 1
            elif shape_type == "rect":
                h = rng.randint(10, 50)
                w = rng.randint(10, 50)
                y1, y2 = max(0, cy - h // 2), min(img_size, cy + h // 2)
                x1, x2 = max(0, cx - w // 2), min(img_size, cx + w // 2)
                mask[y1:y2, x1:x2] = 1
            else:
                # Irregular shape using random walk
                points = [(cy, cx)]
                for _ in range(rng.randint(50, 200)):
                    dy, dx = rng.randint(-2, 3, size=2)
                    ny, nx = points[-1][0] + dy, points[-1][1] + dx
                    ny = np.clip(ny, 0, img_size - 1)
                    nx = np.clip(nx, 0, img_size - 1)
                    points.append((ny, nx))
                    # Fill small area around point
                    r = rng.randint(2, 6)
                    yy, xx = np.ogrid[:img_size, :img_size]
                    blob = ((xx - nx) ** 2 + (yy - ny) ** 2) <= r ** 2
                    mask[blob] = 1

        # ── Apply deforestation spectral signature to image
        deforested = mask > 0
        if deforested.any():
            n_def = deforested.sum()
            image[0][deforested] = rng.uniform(0.08, 0.15, n_def)  # More blue
            image[1][deforested] = rng.uniform(0.06, 0.10, n_def)  # Less green
            image[2][deforested] = rng.uniform(0.10, 0.20, n_def)  # More red (bare soil)
            if num_bands > 3:
                image[3][deforested] = rng.uniform(0.05, 0.15, n_def)  # Less NIR

        # Add sensor noise
        noise = rng.normal(0, 0.005, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)

        # ── Save files
        img_filename = f"sample_{i:05d}.tif"
        mask_filename = f"sample_{i:05d}_mask.tif"

        if HAS_RASTERIO:
            # Save as proper GeoTIFF
            transform = from_bounds(
                -60.0 + rng.uniform(-5, 5),
                -5.0 + rng.uniform(-5, 5),
                -59.0 + rng.uniform(-5, 5),
                -4.0 + rng.uniform(-5, 5),
                img_size, img_size
            )

            # Save image
            with rasterio.open(
                os.path.join(output_dir, img_filename),
                "w", driver="GTiff",
                height=img_size, width=img_size,
                count=num_bands, dtype="float32",
                crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(image)

            # Save mask
            with rasterio.open(
                os.path.join(output_dir, mask_filename),
                "w", driver="GTiff",
                height=img_size, width=img_size,
                count=1, dtype="uint8",
                crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(mask[np.newaxis, ...])
        else:
            # Save as numpy files if rasterio is not available
            np.save(os.path.join(output_dir, f"sample_{i:05d}.npy"), image)
            np.save(os.path.join(output_dir, f"sample_{i:05d}_mask.npy"), mask)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    format_str = "GeoTIFF" if HAS_RASTERIO else "NumPy (.npy)"
    print(f"\n[✓] Generated {num_samples} synthetic samples ({format_str})")
    print(f"    Location: {output_dir}")


# ─── Print Download Instructions ────────────────────────────────────────────────

def print_download_instructions():
    """Print manual download instructions for users without azcopy."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            MultiEarth 2023 Dataset Download                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option 1: Install azcopy (recommended)                      ║
║  ─────────────────────────────────────                       ║
║  1. Download azcopy from:                                    ║
║     https://learn.microsoft.com/azure/storage/               ║
║     common/storage-use-azcopy-v10                            ║
║                                                              ║
║  2. Run this script again:                                   ║
║     python -m src.download_data                              ║
║                                                              ║
║  Option 2: Manual download                                   ║
║  ─────────────────────────────                               ║
║  1. Visit the MultiEarth 2023 challenge page:                ║
║     https://sites.google.com/view/rainforest-challenge/      ║
║     multiearth-2023                                          ║
║                                                              ║
║  2. Follow the download instructions there                   ║
║                                                              ║
║  3. Place the data in: data/multiearth/                      ║
║                                                              ║
║  Option 3: Use synthetic data                                ║
║  ────────────────────────────                                ║
║  python -m src.download_data --generate_synthetic            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download MultiEarth 2023 dataset or generate synthetic data"
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save dataset")
    parser.add_argument("--generate_synthetic", action="store_true",
                        help="Generate synthetic data instead of downloading")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size for synthetic data")
    args = parser.parse_args()

    if args.generate_synthetic:
        generate_synthetic_geotiffs(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            img_size=args.img_size,
        )
    else:
        if check_azcopy():
            success = download_with_azcopy(args.output_dir)
            if not success:
                print("\n[!] Download failed. Generating synthetic data instead...")
                generate_synthetic_geotiffs(output_dir=args.output_dir)
        else:
            print("[!] azcopy is not installed on your system.\n")
            print_download_instructions()
            print("[*] Generating synthetic data as fallback...")
            generate_synthetic_geotiffs(output_dir=args.output_dir,
                                        num_samples=args.num_samples)


if __name__ == "__main__":
    main()
