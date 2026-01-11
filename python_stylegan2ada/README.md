# Python StyleGAN2-ADA for Synthetic Image Generation

**Professional-grade synthetic image generation using NVIDIA's StyleGAN2-ADA.**

This Python implementation provides significantly better results than DCGAN for small datasets (~70 images). StyleGAN2-ADA uses adaptive augmentation to prevent overfitting, making it ideal for limited training data.

---

## 🎯 Why StyleGAN2-ADA?

| Feature | MATLAB DCGAN | Python StyleGAN2-ADA |
|---------|--------------|---------------------|
| Small dataset performance | Moderate | **Excellent** |
| Image quality | Good | **Outstanding** |
| Training stability | Requires tuning | **Self-adaptive** |
| Augmentation | Manual | **Automatic (ADA)** |
| Resolution support | Limited | **Up to 1024x1024** |
| Diversity control | Limited | **Fine-grained (truncation)** |

---

## 📁 Directory Structure

```
python_stylegan2ada/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── preprocess.py                 # Image preprocessing (crop, resize)
│   ├── train.py                      # Training wrapper
│   └── generate.py                   # Image generation
├── data/
│   └── images/                       # Put your training images here
├── outputs/
│   ├── models/                       # Trained models (.pkl files)
│   ├── samples/                      # Training samples
│   └── synthetic/                    # Generated synthetic images
└── stylegan2-ada-pytorch/            # Official NVIDIA repo (clone this)
```

---

## 🚀 Quick Start (One Evening Setup)

### Step 1: Prerequisites

**Hardware:**
- NVIDIA GPU with 8+ GB VRAM (e.g., RTX A2000)
- CUDA 11.3+ and compatible drivers

**Software:**
- Python 3.8+
- CUDA toolkit installed

### Step 2: Verify GPU

```bash
# Check if GPU is visible
nvidia-smi

# Check CUDA in Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected output:**
```
CUDA: True, Device: NVIDIA RTX A2000
```

**Troubleshooting:**
- If `CUDA: False`:
  - Check CUDA installation: `nvcc --version`
  - Reinstall PyTorch with correct CUDA version (see Step 3)
  - Update NVIDIA drivers
- If GPU not found:
  - Verify `nvidia-smi` works
  - Check that GPU is not disabled in BIOS
  - Reboot after driver installation

### Step 3: Install Dependencies

```bash
# Create virtual environment (recommended)
cd python_stylegan2ada
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

**CUDA Version Check:**
- Run `nvcc --version` to see your CUDA version
- Visit https://pytorch.org to get the correct install command for your CUDA version

### Step 4: Clone StyleGAN2-ADA Repository

```bash
cd python_stylegan2ada
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd ..
```

### Step 5: Prepare Your Dataset

**Option A: Use original large images (4200x2128)**

```bash
# 1. Place your ~70 images in data/images/
cp /path/to/your/images/*.jpg python_stylegan2ada/data/images/

# 2. Preprocess with auto-crop (removes white background)
python scripts/preprocess.py \
    --input python_stylegan2ada/data/images \
    --output python_stylegan2ada/data/images/processed \
    --resolution 256 128 \
    --threshold 240 \
    --padding 20

# This creates processed images at 256x128 (preserves ~2:1 aspect ratio)
```

**Why 256x128?** Your images have ~2:1 aspect ratio (4200x2128). Using 256x128 preserves this ratio, avoiding distortion. If you prefer square images, use `--resolution 256 256`.

**Option B: Already preprocessed images**

```bash
# Just copy them to data/images/
cp /path/to/preprocessed/*.png python_stylegan2ada/data/images/
```

**Preprocessing Options:**
- `--threshold 240`: Adjust for background color (lower = detect darker backgrounds)
- `--padding 20`: Add space around detected object
- `--keep-aspect`: Pad to maintain aspect ratio (creates white borders)
- `--no-autocrop`: Skip automatic cropping

### Step 6: Train the Model

```bash
# Basic training with optimal defaults for ~70 images
python scripts/train.py \
    --data python_stylegan2ada/data/images/processed \
    --outdir python_stylegan2ada/outputs/models \
    --gpus 1 \
    --batch 8 \
    --kimg 2000 \
    --snap 10

# Training will take ~2-4 hours on RTX A2000
```

**What happens during training:**
- Models saved every 10 ticks (`--snap 10`) in `outputs/models/`
- Training progress logged to console and TensorBoard
- Automatic augmentation (ADA) prevents overfitting
- Sample images generated in training folder

**If you run out of VRAM:**
```bash
# Reduce batch size
python scripts/train.py --data python_stylegan2ada/data/images/processed --batch 4

# Or even smaller
python scripts/train.py --data python_stylegan2ada/data/images/processed --batch 2
```

**Monitor training:**
```bash
# In another terminal
tensorboard --logdir python_stylegan2ada/outputs/models
# Open http://localhost:6006 in browser
```

### Step 7: Generate Synthetic Images

```bash
# Generate 2000 synthetic images from latest model
python scripts/generate.py --num 2000

# Images saved in outputs/synthetic/ as synthetic_000000.png, synthetic_000001.png, ...
```

**Advanced generation:**
```bash
# Use specific model
python scripts/generate.py \
    --model python_stylegan2ada/outputs/models/00000-custom-auto1/network-snapshot-000200.pkl \
    --num 500

# Control diversity with truncation (0.5 = less diverse, 0.9 = more diverse)
python scripts/generate.py --num 100 --truncation 0.5

# Generate preview grid
python scripts/generate.py --num 16 --grid --outdir python_stylegan2ada/outputs/samples
```

---

## 📊 Training Parameters Explained

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to training images |
| `--outdir` | `outputs/models` | Where to save models |
| `--gpus` | 1 | Number of GPUs |
| `--batch` | 8 | Batch size (reduce if VRAM error) |
| `--kimg` | 2000 | Training length (thousands of images) |
| `--snap` | 10 | Model save interval |

### Augmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aug` | `ada` | Adaptive augmentation (best for small data) |
| `--mirror` | 1 | Horizontal flipping (doubles dataset) |

### Understanding --kimg

The `--kimg` parameter controls training duration:
- **For ~70 images**: `--kimg 2000` trains for ~2-4 hours
- **Rule of thumb**: Higher kimg = better quality (up to a point)
- **Too high**: Risk of overfitting even with ADA
- **Too low**: Model doesn't converge

**Recommended kimg values:**
- 50-100 images: `--kimg 1500-2500`
- 100-200 images: `--kimg 2500-4000`
- 200+ images: `--kimg 4000-6000`

### Batch Size vs VRAM

| GPU VRAM | Max Batch (256x128) | Max Batch (256x256) | Max Batch (512x512) |
|----------|---------------------|---------------------|---------------------|
| 4 GB | 4 | 2 | 1 |
| 8 GB (A2000) | 16 | 8 | 2 |
| 12 GB | 32 | 16 | 4 |
| 24 GB | 64 | 32 | 8 |

**If you get "Out of Memory" error:**
1. Reduce batch size: `--batch 4` or `--batch 2`
2. Close other GPU applications
3. Use smaller resolution during preprocessing

---

## 🔧 Troubleshooting

### GPU Issues

**Problem: `torch.cuda.is_available()` returns `False`**

Solutions:
```bash
# 1. Check CUDA installation
nvcc --version

# 2. Check GPU drivers
nvidia-smi

# 3. Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Verify installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Problem: CUDA version mismatch**

```bash
# Check CUDA versions
nvcc --version              # CUDA toolkit version
python3 -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version

# These should match (11.8, 12.1, etc.)
# If not, reinstall PyTorch matching your CUDA toolkit
```

**Problem: "CUDA out of memory"**

```bash
# Solution 1: Reduce batch size
python scripts/train.py --data [...] --batch 4

# Solution 2: Kill other GPU processes
nvidia-smi  # Find PIDs using GPU
kill -9 <PID>

# Solution 3: Use smaller resolution
python scripts/preprocess.py --resolution 128 64
```

### Training Issues

**Problem: Training diverges (loss becomes NaN)**

Solutions:
- Use default `--cfg auto` (don't override)
- Enable `--mirror 1` for data augmentation
- Reduce learning rate: `--gamma 10` (higher gamma = lower LR)

**Problem: Generated images look identical (mode collapse)**

Solutions:
- Train longer: increase `--kimg`
- Check that `--aug ada` is enabled
- Use `--mirror 1` to double dataset size
- Ensure diverse training data

**Problem: Training is very slow**

```bash
# Check GPU utilization
nvidia-smi -l 1

# If GPU utilization is low:
# 1. Increase batch size (if VRAM allows)
# 2. Check that data loading isn't bottleneck
# 3. Use faster storage (SSD)
```

### Generation Issues

**Problem: Generated images have artifacts**

Solutions:
- Try different truncation: `--truncation 0.5` to `0.9`
- Use later model checkpoint (higher kimg)
- Generate from different seeds: `--seeds 1000-1999`

**Problem: Images not diverse enough**

Solutions:
- Increase truncation: `--truncation 0.9`
- Check training data diversity
- Train longer: higher `--kimg`

---

## 📈 Expected Results Timeline

**Typical training progression (70 images, RTX A2000):**

| Time | Kimg | Quality |
|------|------|---------|
| 30 min | 500 | Blurry shapes, colors recognizable |
| 1 hour | 1000 | Basic structure visible, some details |
| 2 hours | 2000 | Good quality, fine details emerging |
| 3 hours | 3000 | High quality, small improvements |
| 4 hours | 4000 | Excellent quality, diminishing returns |

**Stop training when:**
- Generated samples look realistic
- Quality stops improving (check sample images)
- Validation metrics plateau

**Continue training if:**
- Images still blurry
- Details missing
- Colors not accurate

---

## 🎨 Advanced Usage

### Resume Training

```bash
# Resume from specific checkpoint
python scripts/train.py \
    --data python_stylegan2ada/data/images/processed \
    --resume python_stylegan2ada/outputs/models/00000-custom-auto1/network-snapshot-000100.pkl
```

### Custom Resolutions

```bash
# Square images (easier for StyleGAN2)
python scripts/preprocess.py --resolution 256 256

# High resolution (requires more VRAM)
python scripts/preprocess.py --resolution 512 512
python scripts/train.py --data [...] --batch 2  # Smaller batch for high-res
```

### Batch Generation Script

```bash
# Generate multiple batches
for i in {0..9}; do
    python scripts/generate.py --num 200 --seeds $((i*200))-$(((i+1)*200-1))
done

# Generates 2000 images in 10 batches
```

---

## 🔬 Understanding the Tech

### Why StyleGAN2-ADA vs DCGAN?

1. **Adaptive Augmentation**: Automatically prevents overfitting on small datasets
2. **Progressive Growing**: Better convergence and stability
3. **Style Mixing**: Creates more diverse outputs
4. **Perceptual Path Length**: Higher quality metric optimization

### Training Data Requirements

**Minimum:** 50-100 images
**Recommended:** 100+ images
**Optimal:** 500+ images

**Data quality > quantity:**
- 70 high-quality, diverse images > 500 similar images
- Remove duplicates and near-duplicates
- Ensure variety in pose, lighting, background

### Model Architecture

StyleGAN2-ADA automatically selects architecture based on dataset:
- **auto (default)**: Best choice for most datasets
- **stylegan2**: Full StyleGAN2 (larger datasets)
- **paper256**: Optimized for 256x256 resolution

---

## 📝 File Naming Conventions

**Training data:** Any image format (`.jpg`, `.png`, `.bmp`)

**Generated images:** `synthetic_XXXXXX.png`
- Sequential numbering: `000000`, `000001`, ...
- Easy to track and organize
- Compatible with most data loading pipelines

**Model checkpoints:** `network-snapshot-XXXXXX.pkl`
- `XXXXXX`: kimg at save time
- Example: `network-snapshot-002000.pkl` = trained for 2000 kimg

---

## 🆚 MATLAB vs Python: Quick Reference

**Switch between implementations:**

```bash
# View MATLAB implementation (main branch)
git checkout main  # or master

# View Python implementation
git checkout python-stylegan2-ada

# Compare branches
git diff main python-stylegan2-ada
```

**When to use MATLAB:**
- You're already familiar with MATLAB
- Integration with existing MATLAB pipeline
- Educational purposes (simpler architecture)

**When to use Python:**
- Best possible image quality
- Small dataset (<100 images)
- Modern GAN architecture
- Production use
- State-of-the-art results

---

## 📚 Additional Resources

**Official StyleGAN2-ADA:**
- Paper: https://arxiv.org/abs/2006.06676
- GitHub: https://github.com/NVlabs/stylegan2-ada-pytorch
- NVlabs: https://github.com/NVlabs

**Training tips:**
- NVIDIA StyleGAN2-ADA blog
- Reddit r/MachineLearning
- Papers With Code: StyleGAN2-ADA

**GPU optimization:**
- PyTorch CUDA docs: https://pytorch.org/docs/stable/cuda.html
- CUDA toolkit: https://developer.nvidia.com/cuda-toolkit

---

## 🐛 Common Errors and Fixes

### Import Error: No module named 'torch'
```bash
pip install torch torchvision
```

### Import Error: No module named 'cv2'
```bash
pip install opencv-python
```

### RuntimeError: CUDA out of memory
```bash
python scripts/train.py --batch 2  # Reduce batch size
```

### FileNotFoundError: stylegan2-ada-pytorch/train.py
```bash
cd python_stylegan2ada
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

### ValueError: not enough images in dataset
```bash
# Make sure you have at least 50 images
ls python_stylegan2ada/data/images/ | wc -l
```

---

## ⚡ Performance Tips

1. **Use SSD for data storage** (not HDD)
2. **Close other GPU applications** during training
3. **Enable mirror augmentation** (`--mirror 1`)
4. **Start with lower resolution** (256x256) then scale up
5. **Monitor with TensorBoard** to catch issues early
6. **Save checkpoints frequently** (`--snap 10`)

---

## 📧 Getting Help

If you encounter issues:

1. Check this README (Troubleshooting section)
2. Verify GPU with `nvidia-smi` and `torch.cuda.is_available()`
3. Check StyleGAN2-ADA GitHub issues
4. Ensure dependencies installed correctly
5. Try with smaller batch size and resolution

---

## 🎓 Citation

If you use this code for research, please cite the original StyleGAN2-ADA paper:

```bibtex
@inproceedings{Karras2020ada,
  title={Training Generative Adversarial Networks with Limited Data},
  author={Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle={Proc. NeurIPS},
  year={2020}
}
```

---

**Ready to generate better synthetic images in one evening!** 🚀

Start with Step 1 and follow the Quick Start guide. You should have high-quality synthetic images within 3-4 hours.
