# NAVODILA ZA RETRAINING - Barvne Busbar Slike

## Problem
Star model je bil treniran s **staro kodo**, ki je:
1. Avtomatsko zaznala slike kot "grayscale"
2. Konvertirala vaše **bakrene RGB slike → grayscale** (rgb2gray)
3. Generirala samo sive slike

## Rešitev
**Nova koda** (commit 1dae37c + a6b545b) **vedno uporablja RGB (3 kanale)** in ne konvertira več v grayscale!

---

## KORAK 1: Dodajte učne slike

```bash
# 1. Kopirajte vaše busbar slike v data/images/
cp /path/to/busbar/*.jpg /home/user/GAN_Busbar/data/images/

# 2. Preverite da so tam (vsaj 50-70 slik)
ls -lh /home/user/GAN_Busbar/data/images/
```

**Vaše busbar slike so PERFEKTNE** - bakrene/zlate tone so **RGB barve**! ✅

---

## KORAK 2A: MATLAB Retraining (priporočeno)

```matlab
% V MATLAB:
cd /home/user/GAN_Busbar

% 1. Preveri učne slike
ls data/images/*.jpg

% 2. Potrdi da je nova koda active
which train_gan  % Naj kaže: /home/user/GAN_Busbar/train_gan.m

% 3. **POBRIŠITE STAR MODEL** (če obstaja)
if exist('outputs/models/generator.mat', 'file')
    delete('outputs/models/generator.mat')
    delete('outputs/models/discriminator.mat')
    fprintf('✓ Deleted old models\n');
end

% 4. Trenirajte nov model z novo kodo
train_gan

% Izpis bo pokazal:
%   Checking if images contain color information...
%   ✓ Found X RGB colored images (out of Y sampled)
%   Color generation: ENABLED - model will generate colored images
```

**Nova koda bo:**
- ✅ Zaznala RGB barve (bakrene tone)
- ✅ Ohranila RGB format (ne konvertira v grayscale)
- ✅ Generirala **barvne busbar slike** z bakrenimi/zlatimi toni!

---

## KORAK 2B: Python StyleGAN2-ADA (boljša kvaliteta)

```bash
cd /home/user/GAN_Busbar

# 1. Preprocess (bo preveril RGB)
python python_stylegan2ada/scripts/preprocess.py \
    --input data/images \
    --output python_stylegan2ada/data/processed

# Izpis bo pokazal:
#   Checking if images contain color information...
#   ✓ Found X RGB colored images (out of Y sampled)
#   Color generation: ENABLED

# 2. Train model
python python_stylegan2ada/scripts/train.py \
    --data python_stylegan2ada/data/processed \
    --gpus 1 \
    --batch 8 \
    --kimg 2000

# 3. Generate images
python python_stylegan2ada/scripts/generate.py \
    --network python_stylegan2ada/outputs/models/XXX/network-snapshot-XXXX.pkl \
    --seeds 0-99 \
    --output outputs/generated_grid.png
```

---

## Zakaj star model ne deluje?

**Star model (treniran pred commit 1dae37c):**
```matlab
% Stara koda - AUTO-DETECT:
if numRGB >= numGray
    params.numChannels = 3;
else
    params.numChannels = 1;  % ← Nastavil 1 kanal!
end

% Potem v preprocessing:
if params.numChannels == 1
    if size(img, 3) == 3
        img = rgb2gray(img);  % ← Konvertiral RGB → grayscale! ❌
    end
end
```

**Nova koda (po commit 1dae37c):**
```matlab
% Nova koda - VEDNO RGB:
params.numChannels = 3;  % ✅ Vedno 3 kanali!

% Preprocessing:
if params.numChannels == 3
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);  % Grayscale → RGB
    end
    % Če je že RGB (3 kanale), pusti kot je! ✅
end
```

---

## Pričakovani rezultati

Po novem treningu z **popravljeno kodo**, generirane slike bodo imele:
- 🟡 **Bakrene/zlate tone** (copper/brass color)
- ⚪ **Belo ozadje** (kot originali)
- 🎨 **RGB barve** (ne več grayscale)

**Generator se bo naučil:**
- Geometrijo busbar komponent (krožne luknje, oblika)
- Bakrene/zlate barve (ne bo več siv)
- Teksturo kovine
- Sence in highlights

---

## Če želite preveriti RGB **PRED** treningom:

```matlab
% Test če so slike RGB:
img = imread('data/images/busbar_001.jpg');
fprintf('Image size: %s\n', mat2str(size(img)));

if size(img, 3) == 3
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);

    % Sample 1000 random pixels
    idx = randperm(numel(R), 1000);
    R_sample = R(idx); G_sample = G(idx); B_sample = B(idx);

    % Check if R, G, B are different
    if isequal(R_sample, G_sample) && isequal(G_sample, B_sample)
        fprintf('❌ FAKE RGB - actually grayscale!\n');
    else
        fprintf('✓ TRUE RGB - colored image (copper/brass tones)!\n');
        fprintf('  R range: [%d, %d]\n', min(R_sample), max(R_sample));
        fprintf('  G range: [%d, %d]\n', min(G_sample), max(G_sample));
        fprintf('  B range: [%d, %d]\n', min(B_sample), max(B_sample));
    end
end
```

Za bakrene/zlate slike, pričakovani rezultat:
```
✓ TRUE RGB - colored image (copper/brass tones)!
  R range: [150, 220]  % Rdeča visoka (bakarna)
  G range: [120, 180]  % Zelena srednja
  B range: [50, 100]   % Modra nizka
```

---

## Summary

1. ❌ **Star problem**: Stara koda je konvertirala RGB→grayscale
2. ✅ **Nova koda**: Vedno ohrani RGB barve
3. 🔄 **Rešitev**: Retrenirati model z novo kodo
4. 🎨 **Rezultat**: Barvne busbar slike z bakrenimi toni!

**Vaše slike so perfektne - samo retrenirati morate z novo kodo!**
