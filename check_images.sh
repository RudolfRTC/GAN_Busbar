#!/bin/bash
# TEST SCRIPT - Run after copying images

echo "=========================================="
echo "Checking if images are ready for training"
echo "=========================================="
echo ""

# Check if images exist
IMAGE_COUNT=$(ls -1 /home/user/GAN_Busbar/data/images/*.{jpg,jpeg,png,bmp} 2>/dev/null | wc -l)

if [ $IMAGE_COUNT -eq 0 ]; then
    echo "❌ ERROR: No images found in ./data/images/"
    echo ""
    echo "You need to copy images from your computer into the Docker container!"
    echo ""
    echo "On YOUR COMPUTER (not in Docker), run:"
    echo ""
    echo "  # 1. Find container ID:"
    echo "  docker ps"
    echo ""
    echo "  # 2. Copy images:"
    echo "  docker cp /path/to/your/busbar/images/. CONTAINER_ID:/home/user/GAN_Busbar/data/images/"
    echo ""
    exit 1
fi

echo "✓ Found $IMAGE_COUNT images"
echo ""

# Check first image for RGB
FIRST_IMAGE=$(ls /home/user/GAN_Busbar/data/images/*.{jpg,jpeg,png,bmp} 2>/dev/null | head -1)

if [ -n "$FIRST_IMAGE" ]; then
    echo "Sample image: $FIRST_IMAGE"

    # Use Python to check if RGB
    python3 << EOF
import cv2
import numpy as np
import sys

try:
    img = cv2.imread('$FIRST_IMAGE')
    if img is None:
        print('❌ Cannot read image')
        sys.exit(1)

    h, w, c = img.shape
    print(f'  Size: {w}x{h}')
    print(f'  Channels: {c}')

    if c == 3:
        # Check if truly RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Sample 100 pixels
        num_pixels = h * w
        sample_size = min(100, num_pixels)
        indices = np.random.choice(num_pixels, sample_size, replace=False)

        flat_img = img_rgb.reshape(-1, 3)
        R = flat_img[indices, 0]
        G = flat_img[indices, 1]
        B = flat_img[indices, 2]

        if np.all(R == G) and np.all(G == B):
            print('  ❌ FAKE RGB (actually grayscale)')
            print('')
            print('  Your images are grayscale!')
            print('  GAN will generate gray images.')
            print('  Use colorization tools to add color.')
        else:
            print('  ✓ TRUE RGB (colored image)')
            print(f'  R range: [{R.min()}, {R.max()}]')
            print(f'  G range: [{G.min()}, {G.max()}]')
            print(f'  B range: [{B.min()}, {B.max()}]')
            print('')
            print('✅ Images are ready for RGB training!')
            print('   GAN will generate COLORED images!')
    else:
        print('  ❌ Grayscale (1 channel)')
        print('')
        print('  Your images are grayscale!')
        print('  GAN will generate gray images.')

except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
EOF

fi

echo ""
echo "=========================================="
echo "Next step: Run MATLAB training"
echo "=========================================="
echo ""
echo "In MATLAB:"
echo "  cd /home/user/GAN_Busbar"
echo "  clear all"
echo "  train_gan"
echo ""
