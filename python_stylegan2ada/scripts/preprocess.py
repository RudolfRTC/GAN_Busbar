#!/usr/bin/env python3
"""
Preprocess images for StyleGAN2-ADA training.

For large images (e.g., 4200x2128) with product on white background:
1. Auto-crop around the object (threshold on white + largest component)
2. Resize to training resolution (default 256x128 to preserve ~2:1 aspect ratio)
3. Save processed images

Usage:
    python scripts/preprocess.py --input data/images --output data/images/processed --resolution 256 128
    python scripts/preprocess.py --input data/images --output data/images/processed --resolution 256 256 --no-autocrop
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def auto_crop_white_background(
    image: np.ndarray,
    threshold: int = 240,
    padding: int = 20
) -> Optional[np.ndarray]:
    """
    Auto-crop image around object by removing white background.

    Args:
        image: Input image (RGB or grayscale)
        threshold: Pixel value threshold for white (0-255)
        padding: Pixels to add around detected object

    Returns:
        Cropped image or None if no object detected
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Create binary mask: 0 for object, 255 for white background
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Warning: No object detected in image")
        return None

    # Find largest contour (assume this is the product)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)

    # Crop image
    if len(image.shape) == 3:
        cropped = image[y_start:y_end, x_start:x_end, :]
    else:
        cropped = image[y_start:y_end, x_start:x_end]

    return cropped


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect: bool = False
) -> np.ndarray:
    """
    Resize image to target resolution.

    Args:
        image: Input image
        target_size: (width, height)
        keep_aspect: If True, pad to maintain aspect ratio

    Returns:
        Resized image
    """
    if keep_aspect:
        # Resize maintaining aspect ratio, then pad
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create white canvas and center image
        if len(image.shape) == 3:
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        else:
            canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas
    else:
        # Direct resize (may distort aspect ratio)
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)


def preprocess_image(
    input_path: Path,
    output_path: Path,
    resolution: Tuple[int, int],
    auto_crop: bool = True,
    threshold: int = 240,
    padding: int = 20,
    keep_aspect: bool = False
) -> bool:
    """
    Preprocess a single image.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Cannot read {input_path}")
            return False

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Auto-crop if requested
        if auto_crop:
            cropped = auto_crop_white_background(image, threshold, padding)
            if cropped is not None:
                image = cropped

        # Resize to target resolution
        resized = resize_image(image, resolution, keep_aspect)

        # Save image
        pil_image = Image.fromarray(resized)
        pil_image.save(output_path, quality=95)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess images for StyleGAN2-ADA training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="python_stylegan2ada/data/images",
        help="Input directory with original images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="python_stylegan2ada/data/images/processed",
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[256, 128],
        metavar=("WIDTH", "HEIGHT"),
        help="Target resolution (width height). Default: 256 128 for 2:1 aspect ratio"
    )
    parser.add_argument(
        "--no-autocrop",
        action="store_true",
        help="Disable automatic cropping of white background"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=240,
        help="White threshold for auto-crop (0-255). Default: 240"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around detected object (pixels). Default: 20"
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Maintain aspect ratio when resizing (pads with white)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        help="Image file extensions to process"
    )

    args = parser.parse_args()

    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    resolution = tuple(args.resolution)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = [ext.lower() for ext in args.extensions]
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))

    if not image_files:
        print(f"Error: No images found in '{input_dir}'")
        print(f"Looked for extensions: {extensions}")
        return 1

    print(f"Found {len(image_files)} images to process")
    print(f"Output resolution: {resolution[0]}x{resolution[1]}")
    print(f"Auto-crop: {not args.no_autocrop}")
    if not args.no_autocrop:
        print(f"  Threshold: {args.threshold}, Padding: {args.padding}")
    print(f"Keep aspect ratio: {args.keep_aspect}")
    print()

    # Process images
    success_count = 0
    for img_path in tqdm(image_files, desc="Processing"):
        output_path = output_dir / img_path.name

        if preprocess_image(
            img_path,
            output_path,
            resolution,
            auto_crop=not args.no_autocrop,
            threshold=args.threshold,
            padding=args.padding,
            keep_aspect=args.keep_aspect
        ):
            success_count += 1

    print(f"\nDone! Successfully processed {success_count}/{len(image_files)} images")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
