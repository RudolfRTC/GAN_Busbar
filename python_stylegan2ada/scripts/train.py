#!/usr/bin/env python3
"""
Train StyleGAN2-ADA model on custom dataset.

This script is a wrapper around NVIDIA's StyleGAN2-ADA-PyTorch.
It provides sensible defaults for small datasets (~70 images).

Prerequisites:
    1. Clone StyleGAN2-ADA repo:
       cd python_stylegan2ada
       git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

    2. Prepare dataset:
       - Place images in data/images/
       - Optionally preprocess: python scripts/preprocess.py

Usage:
    # Basic training with default settings (optimized for ~70 images)
    python scripts/train.py --data python_stylegan2ada/data/images

    # Custom settings
    python scripts/train.py \\
        --data python_stylegan2ada/data/images/processed \\
        --outdir python_stylegan2ada/outputs/models \\
        --gpus 1 \\
        --batch 8 \\
        --kimg 2000 \\
        --snap 10

    # Resume training
    python scripts/train.py --data python_stylegan2ada/data/images --resume python_stylegan2ada/outputs/models/00000-custom-auto1/network-snapshot-000100.pkl
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def check_stylegan2_repo():
    """Check if StyleGAN2-ADA repo is cloned."""
    repo_path = Path("python_stylegan2ada/stylegan2-ada-pytorch")
    train_script = repo_path / "train.py"

    if not train_script.exists():
        print("=" * 80)
        print("ERROR: StyleGAN2-ADA-PyTorch repository not found!")
        print("=" * 80)
        print()
        print("Please clone the repository first:")
        print()
        print("  cd python_stylegan2ada")
        print("  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git")
        print()
        print("Then run this script again.")
        print("=" * 80)
        return False

    return True


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("WARNING: No GPU detected. Training will be very slow!")
            print("Check your CUDA installation and GPU drivers.")
            return False
    except ImportError:
        print("WARNING: PyTorch not installed. Cannot check GPU.")
        return False


def validate_dataset(data_path: Path):
    """Validate dataset directory."""
    if not data_path.exists():
        print(f"ERROR: Dataset directory not found: {data_path}")
        return False

    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(data_path.glob(f"*{ext}")))
        images.extend(list(data_path.glob(f"*{ext.upper()}")))

    if len(images) == 0:
        print(f"ERROR: No images found in {data_path}")
        print(f"Looked for extensions: {image_extensions}")
        return False

    print(f"Dataset: {len(images)} images found in {data_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train StyleGAN2-ADA on custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Essential arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset directory (folder with images)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="python_stylegan2ada/outputs/models",
        help="Output directory for models and logs. Default: python_stylegan2ada/outputs/models"
    )

    # Training configuration
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs. Default: 1"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Total batch size (will be distributed across GPUs). Default: 8 (good for RTX A2000 8GB)"
    )
    parser.add_argument(
        "--kimg",
        type=int,
        default=2000,
        help="Total training length in thousands of images. Default: 2000 (~2-3 hours on RTX A2000)"
    )
    parser.add_argument(
        "--snap",
        type=int,
        default=10,
        help="Snapshot interval (save model every N ticks). Default: 10"
    )

    # Augmentation
    parser.add_argument(
        "--aug",
        type=str,
        default="ada",
        choices=["noaug", "ada", "fixed"],
        help="Augmentation mode. Default: ada (adaptive, best for small datasets)"
    )
    parser.add_argument(
        "--mirror",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable horizontal flipping. Default: 1 (enabled)"
    )

    # Model configuration
    parser.add_argument(
        "--cfg",
        type=str,
        default="auto",
        help="Model config. Default: auto (automatically select based on dataset)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="R1 regularization weight. Default: None (auto)"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (.pkl file)"
    )

    # Advanced options
    parser.add_argument(
        "--freezed",
        type=int,
        default=None,
        help="Freeze-D: freeze first layers of discriminator. Default: None (no freezing)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fid50k_full"],
        help="Quality metrics to compute. Default: fid50k_full"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("StyleGAN2-ADA Training Script")
    print("=" * 80)
    print()

    # Check prerequisites
    if not check_stylegan2_repo():
        return 1

    print()
    check_gpu()
    print()

    # Validate dataset
    data_path = Path(args.data)
    if not validate_dataset(data_path):
        return 1

    print()
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Dataset:      {args.data}")
    print(f"Output:       {args.outdir}")
    print(f"GPUs:         {args.gpus}")
    print(f"Batch size:   {args.batch}")
    print(f"Training:     {args.kimg}k images")
    print(f"Augmentation: {args.aug}")
    print(f"Mirror:       {args.mirror}")
    print(f"Config:       {args.cfg}")
    if args.resume:
        print(f"Resume from:  {args.resume}")
    print()

    # Build command
    cmd = [
        sys.executable,
        "python_stylegan2ada/stylegan2-ada-pytorch/train.py",
        "--outdir", args.outdir,
        "--data", args.data,
        "--gpus", str(args.gpus),
        "--batch", str(args.batch),
        "--kimg", str(args.kimg),
        "--snap", str(args.snap),
        "--aug", args.aug,
        "--mirror", str(args.mirror),
        "--cfg", args.cfg,
    ]

    # Add optional arguments
    if args.gamma is not None:
        cmd.extend(["--gamma", str(args.gamma)])

    if args.resume is not None:
        cmd.extend(["--resume", args.resume])

    if args.freezed is not None:
        cmd.extend(["--freezed", str(args.freezed)])

    if args.metrics:
        cmd.extend(["--metrics", ",".join(args.metrics)])

    # Print command
    print("Executing command:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("Dry run mode - not executing")
        return 0

    print("=" * 80)
    print("Starting training...")
    print("=" * 80)
    print()

    # Execute training
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 80)
        print("Training completed successfully!")
        print(f"Models saved in: {args.outdir}")
        print("=" * 80)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print(f"Training failed with error code {e.returncode}")
        print("=" * 80)
        return e.returncode
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
