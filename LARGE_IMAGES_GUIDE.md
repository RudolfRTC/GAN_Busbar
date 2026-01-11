# Guide: Working with Large Images (4200x2128)

## Problem
Your original images are **4200x2128 pixels** (aspect ratio ≈ 2:1). Training a GAN at this resolution is:
- Memory intensive (will crash on laptop GPU)
- Computationally expensive
- Unnecessary for learning the object structure

## Solution: trainSize Parameter

The code now uses **`params.trainSize`** instead of a square `imageSize`:

```matlab
params.trainSize = [64 128];  % [height width]
```

### Supported Sizes

| trainSize    | Aspect Ratio | Memory Usage | Recommended For           |
|--------------|--------------|--------------|---------------------------|
| `[64 128]`   | 2:1          | Low          | **Laptop GPU (DEFAULT)**  |
| `[128 256]`  | 2:1          | Medium       | Desktop GPU, higher quality |
| `[64 64]`    | 1:1          | Low          | Fast training, square output |
| `[128 128]`  | 1:1          | Medium       | Balanced, square output   |
| `[256 256]`  | 1:1          | High         | High quality, requires strong GPU |

### Recommendation for Your 4200x2128 Images

**Use `[64 128]` (default):**
- ✅ Preserves 2:1 aspect ratio
- ✅ Memory efficient for laptop GPU
- ✅ Auto-crop removes white background first
- ✅ Batch size = 8 fits in memory

**If you have more GPU memory, use `[128 256]`:**
- Better quality
- Still preserves aspect ratio
- May need to reduce batch size

## How It Works

### 1. Auto-Crop (Optional, enabled by default)
```matlab
params.autoCrop = true;           % Enable auto-crop
params.cropThreshold = 0.85;      % White threshold (0-1)
```

**Process:**
- Detects white background (> 85% brightness)
- Finds largest object (your industrial part)
- Crops with 10% padding
- **Result:** 4200x2128 → ~1500x750 (varies per image)

### 2. Resize to trainSize
```matlab
params.trainSize = [64 128];
```

**Process:**
- Resizes cropped image to [64 x 128]
- Preserves aspect ratio during resize
- **Result:** 1500x750 → 64x128

### 3. Training
- Generator learns on 64x128 images
- Much faster and memory efficient
- Still captures object features

### 4. Synthetic Output
- Generated images are 64x128
- Can be upscaled later if needed using:
  ```matlab
  imgUpscaled = imresize(imgGenerated, [2128 4200]);
  ```

## Usage

### 1. Edit train_gan.m

```matlab
%% ===== PARAMETERS (MODIFY HERE) =====
params = struct();

% Data parameters
params.dataFolder = './data/images';
params.trainSize = [64 128];         % ← CHANGE THIS if needed
                                     %   [64 128]  - laptop GPU
                                     %   [128 256] - desktop GPU
params.autoCrop = true;              % Keep enabled for white background
params.cropThreshold = 0.85;         % Adjust if crop is too aggressive

% Training parameters
params.miniBatchSize = 8;            % ← Reduce to 4-6 if GPU memory error
params.numEpochs = 300;
```

### 2. Run

```matlab
train_gan
```

**Expected output:**
```
Parameters:
  Training Size: 64x128 (HxW)
  Aspect Ratio: 2.00:1
  Latent Dim: 100
  Epochs: 300
  Batch Size: 8
  ...

Building Generator...
  Generator built successfully:
    Input shape:  [1 x 1 x 100 x N]
    Output shape: [64 x 128 x 3 x N]
  Generator parameters: 5243075

Sanity check: Testing generator...
  Generator output shape: [64 x 128 x 3 x 2]
  Expected shape:         [64 x 128 x 3 x 2]
  ✓ Generator test PASSED!
```

## Troubleshooting

### GPU Out of Memory

**Error:**
```
Out of memory on device. To view more detail about available memory on the GPU, use 'gpuDevice()'
```

**Solutions:**
1. Reduce batch size:
   ```matlab
   params.miniBatchSize = 4;  % or even 2
   ```

2. Use smaller trainSize:
   ```matlab
   params.trainSize = [64 64];  % Square, smaller
   ```

3. Use CPU (slower):
   ```matlab
   params.executionEnvironment = 'cpu';
   ```

### Auto-Crop Too Aggressive

If objects are being cut off:

```matlab
params.cropThreshold = 0.75;  % Lower = less aggressive
% or
params.autoCrop = false;       % Disable entirely
```

Then manually crop your images before training.

### Generator Shape Error

**Error:**
```
Layer 'tconv1': Input data must have two spatial dimensions and one channel dimension.
```

**This should be fixed now!** But if you still see it:
- Check that `buildGenerator.m` uses `imageInputLayer([1 1 latentDim])`
- Verify Z is generated as: `randn(1, 1, params.latentDim, N)`

## After Training

### Upscale Synthetic Images

If you need 4200x2128 output:

```matlab
% Load generator
load('./outputs/models/generator.mat', 'netG', 'params');

% Generate one image
Z = dlarray(randn(1, 1, params.latentDim, 1, 'single'), 'SSCB');
XGenerated = predict(netG, Z);
XGenerated = extractdata(XGenerated);

% Denormalize [-1,1] → [0,1]
img = (XGenerated(:,:,:,1) + 1) / 2;

% Upscale to original resolution
imgUpscaled = imresize(img, [2128 4200]);

% Save
imwrite(imgUpscaled, 'synthetic_upscaled.png');
```

### Use for Data Augmentation

Combine original + synthetic images:

```matlab
% Place synthetic images in:
./data/images_augmented/

% Copy originals:
copyfile('./data/images/*.jpg', './data/images_augmented/');

% Copy synthetic:
copyfile('./outputs/synthetic/*.png', './data/images_augmented/');

% Now you have ~70 original + 2000 synthetic = 2070 images for training!
```

## Memory Usage Reference

| trainSize    | Batch Size | Approx GPU Memory | Can Train On          |
|--------------|------------|-------------------|-----------------------|
| [64 128]     | 8          | ~2 GB             | Most laptop GPUs      |
| [64 128]     | 16         | ~3 GB             | Mid-range laptop GPU  |
| [128 256]    | 8          | ~4 GB             | Desktop GPU           |
| [128 256]    | 16         | ~6 GB             | High-end GPU          |
| [256 256]    | 8          | ~8 GB             | High-end GPU only     |

## Summary

✅ **Use `trainSize = [64 128]`** for your 4200x2128 images
✅ **Enable auto-crop** to remove white background
✅ **Start with batch size 8** for laptop GPU
✅ **Sanity check** runs automatically to verify shapes
✅ **Upscale later** if you need full resolution

This approach balances quality, memory efficiency, and training speed!
