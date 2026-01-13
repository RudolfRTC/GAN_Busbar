#!/usr/bin/env python3
"""
Generate synthetic images using trained StyleGAN2-ADA model.

This script loads a trained .pkl model and generates synthetic images.

Prerequisites:
    - Trained model (.pkl file) in outputs/models/
    - StyleGAN2-ADA repo cloned in python_stylegan2ada/

Usage:
    # Generate 2000 images from latest model
    python scripts/generate.py --num 2000

    # Generate from specific model
    python scripts/generate.py --model python_stylegan2ada/outputs/models/00000-custom-auto1/network-snapshot-000100.pkl --num 500

    # Generate with specific seed range
    python scripts/generate.py --seeds 0-999

    # Generate grid of sample images
    python scripts/generate.py --num 16 --grid --outdir python_stylegan2ada/outputs/samples
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def find_latest_model(models_dir: Path) -> Optional[Path]:
    """Find the latest .pkl model in the output directory."""
    pkl_files = list(models_dir.glob("**/*.pkl"))

    if not pkl_files:
        return None

    # Sort by modification time
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pkl_files[0]


def load_model(model_path: Path, device: str = "cuda"):
    """Load StyleGAN2 model from .pkl file."""
    print(f"Loading model from: {model_path}")

    # Add StyleGAN2 repo to path
    stylegan2_path = Path("python_stylegan2ada/stylegan2-ada-pytorch")
    if not stylegan2_path.exists():
        raise RuntimeError(
            "StyleGAN2-ADA repo not found. Please clone it:\n"
            "  cd python_stylegan2ada\n"
            "  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git"
        )

    sys.path.insert(0, str(stylegan2_path))

    try:
        with open(model_path, 'rb') as f:
            G = torch.load(f, map_location=device)
            if isinstance(G, dict):
                G = G['G_ema']  # Extract generator if saved as dict
        print(f"Model loaded successfully")
        print(f"Output resolution: {G.img_resolution}x{G.img_resolution}")
        print(f"Latent dimension: {G.z_dim}")
        return G
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def generate_images(
    G,
    seeds: List[int],
    output_dir: Path,
    device: str = "cuda",
    truncation_psi: float = 0.7
):
    """Generate images from seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(seeds)} images...")
    print(f"Truncation psi: {truncation_psi}")
    print(f"Output directory: {output_dir}")
    print()

    for seed in tqdm(seeds, desc="Generating"):
        # Generate latent vector
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z).to(device)

        # Generate image
        with torch.no_grad():
            img = G(z, None, truncation_psi=truncation_psi)

        # Convert to PIL image
        img = (img + 1) * 127.5  # [-1, 1] -> [0, 255]
        img = img.clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray(img)

        # Save image
        output_path = output_dir / f"synthetic_{seed:06d}.png"
        pil_img.save(output_path)


def generate_grid(
    G,
    seeds: List[int],
    output_path: Path,
    grid_size: Optional[tuple] = None,
    device: str = "cuda",
    truncation_psi: float = 0.7
):
    """Generate a grid of images."""
    print(f"Generating grid with {len(seeds)} images...")

    # Determine grid size
    if grid_size is None:
        grid_w = int(np.ceil(np.sqrt(len(seeds))))
        grid_h = int(np.ceil(len(seeds) / grid_w))
    else:
        grid_w, grid_h = grid_size

    # Generate images
    images = []
    for seed in tqdm(seeds, desc="Generating"):
        z = np.random.RandomState(seed).randn(1, G.z_dim)
        z = torch.from_numpy(z).to(device)

        with torch.no_grad():
            img = G(z, None, truncation_psi=truncation_psi)

        img = (img + 1) * 127.5
        img = img.clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()

        # FIXED: Convert grayscale to RGB if needed
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # [H, W, 1] -> [H, W, 3]

        images.append(img)

    # Create grid
    img_h, img_w = images[0].shape[:2]
    grid = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_w
        col = idx % grid_w
        grid[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = img

    # Save grid
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"Grid saved to: {output_path}")


def parse_seed_range(seed_str: str) -> List[int]:
    """Parse seed specification (e.g., '0-999', '0,1,2', '42')."""
    if '-' in seed_str:
        # Range: '0-999'
        start, end = seed_str.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in seed_str:
        # List: '0,1,2'
        return [int(s) for s in seed_str.split(',')]
    else:
        # Single seed
        return [int(seed_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with StyleGAN2-ADA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .pkl model. If not specified, uses latest model from outputs/models/"
    )

    # Generation options
    parser.add_argument(
        "--num",
        type=int,
        default=2000,
        help="Number of images to generate. Default: 2000"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seed specification: '0-999', '0,1,2', or '42'. Overrides --num if specified."
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="Truncation psi (0-1). Lower = more typical, higher = more diverse. Default: 0.7"
    )

    # Output options
    parser.add_argument(
        "--outdir",
        type=str,
        default="python_stylegan2ada/outputs/synthetic",
        help="Output directory for generated images. Default: python_stylegan2ada/outputs/synthetic"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Generate image grid instead of individual files"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Grid size (width height). Default: auto"
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu). Default: auto-detect"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("StyleGAN2-ADA Image Generation")
    print("=" * 80)
    print()

    # Check GPU
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA requested but not available, using CPU")
            args.device = "cpu"
    else:
        print("Using CPU (will be slow)")
    print()

    # Find model
    if args.model is None:
        models_dir = Path("python_stylegan2ada/outputs/models")
        model_path = find_latest_model(models_dir)
        if model_path is None:
            print(f"ERROR: No model found in {models_dir}")
            print("Please train a model first or specify --model")
            return 1
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            return 1

    # Load model
    try:
        G = load_model(model_path, args.device)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print()

    # Determine seeds
    if args.seeds is not None:
        seeds = parse_seed_range(args.seeds)
    else:
        seeds = list(range(args.num))

    # Generate images
    output_dir = Path(args.outdir)

    if args.grid:
        # Generate grid
        grid_path = output_dir / "grid.png"
        generate_grid(
            G,
            seeds,
            grid_path,
            tuple(args.grid_size) if args.grid_size else None,
            args.device,
            args.truncation
        )
    else:
        # Generate individual images
        generate_images(G, seeds, output_dir, args.device, args.truncation)

    print()
    print("=" * 80)
    print("Generation complete!")
    print(f"Images saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
