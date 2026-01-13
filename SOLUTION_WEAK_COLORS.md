# REŠITEV ZA ŠIBKE/TEMNE BARVE

## Problem identificiran! ✅

Vaše generirane slike **SO RGB** (R≠G≠B) - moj kod **JE PRAVILEN**!

**AMPAK:** Barve so **PRETEMNE** in **PREŠIBKE**!

---

## 📊 Primerjava:

| | R | G | B | R-B razlika | Svetlost |
|---|---|---|---|-------------|----------|
| **Učne slike** | 205 | 193 | 175 | **30** (močna bakrna) | **191** (svetlo) |
| **Generirane** | 109 | 94 | 89 | **20** (šibka bakrna) | **97** (temno) |

**Generator outputa:**
- ❌ Samo **50% svetlosti** originala (97 vs 191)
- ❌ **Šibkejše barve** (R-B = 20 vs 30)
- ❌ **Ozek rang** tanh vrednosti (okoli 0, ne -1 do +1)

---

## 🎯 VZROK:

**Generator NI DOVOLJ TRENIRAN!**

Generator se je naučil **geometrije** (oblike busbar), ampak še **NI** se naučil:
- Močnih barv (visok kontrast R vs B)
- Svetlosti (visoke vrednosti)
- Polnega ranga tanh outputa [-1, +1]

**TO NI BUG V KODI - TO JE TRAINING PROBLEM!**

300 epoch **NI DOVOLJ** za majhen dataset (350 slik)!

---

## ✅ REŠITEV 1: Več Treninga (PRIPOROČENO)

### Spremenite train_gan.m:

```matlab
% Training parameters
params.numEpochs = 600;              % Povečano iz 300 → 600
params.miniBatchSize = 8;
params.learnRate = 0.0002;
params.beta1 = 0.5;
```

**Pričakovani rezultati:**
- **Epoch 300**: Geometrija OK, barve šibke (trenutno stanje)
- **Epoch 400-500**: Barve se začnejo krepiti
- **Epoch 600**: Močne bakrene barve, svetlejše slike

**Traja:** ~2-4 ure več treninga

---

## ✅ REŠITEV 2: Balance G vs D Learning Rates

Generator potrebuje **več "confidence"** da outputa močnejše vrednosti.

### Spremenite train_gan.m vrstice 165-166:

**STARO:**
```matlab
avgGradientsG = [];
avgGradientsSquaredG = [];
avgGradientsD = [];
avgGradientsSquaredD = [];
```

**NOVO - Dodajte različne learning rate:**
```matlab
% Different learning rates for G and D
learnRateG = 0.0003;  % Generator faster (was 0.0002)
learnRateD = 0.0001;  % Discriminator slower (was 0.0002)

avgGradientsG = [];
avgGradientsSquaredG = [];
avgGradientsD = [];
avgGradientsSquaredD = [];
```

**In spremenite update-e (vrstice ~220, ~230):**

**STARO:**
```matlab
[netG.Learnables, avgGradientsG, avgGradientsSquaredG] = ...
    adamupdate(netG.Learnables, gradG, avgGradientsG, avgGradientsSquaredG, ...
               iteration, params.learnRate, params.beta1);
```

**NOVO:**
```matlab
[netG.Learnables, avgGradientsG, avgGradientsSquaredG] = ...
    adamupdate(netG.Learnables, gradG, avgGradientsG, avgGradientsSquaredG, ...
               iteration, learnRateG, params.beta1);  % Use learnRateG
```

Isto za Discriminator (use `learnRateD`).

---

## ✅ REŠITEV 3: Post-Processing Brightness Fix (HITRO)

Če ne želite retrenirati, lahko **povečate svetlost** že generiranih slik:

### Ustvarite brighten_images.m:

```matlab
function brighten_images()
    % Brighten synthetic images to match training data brightness

    inputFolder = './outputs/synthetic';
    outputFolder = './outputs/synthetic_brightened';
    mkdir(outputFolder);

    imageFiles = dir(fullfile(inputFolder, '*.png'));

    % Target: učne slike imajo povprečno svetlost 191
    % Generirane: 97
    % Faktor: 191/97 = 1.97

    brightenFactor = 1.95;  % Povečaj 95%
    colorBoost = 1.3;       % Okrepi barve 30%

    for i = 1:numel(imageFiles)
        % Load
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        img = imread(imgPath);
        img = double(img);

        % Convert to HSV
        img_hsv = rgb2hsv(img / 255);

        % Increase Value (brightness)
        img_hsv(:,:,3) = min(1, img_hsv(:,:,3) * brightenFactor);

        % Increase Saturation (color strength)
        img_hsv(:,:,2) = min(1, img_hsv(:,:,2) * colorBoost);

        % Convert back to RGB
        img_bright = hsv2rgb(img_hsv);
        img_bright = uint8(img_bright * 255);

        % Save
        outputPath = fullfile(outputFolder, imageFiles(i).name);
        imwrite(img_bright, outputPath);

        if mod(i, 100) == 0
            fprintf('Processed %d/%d\n', i, numel(imageFiles));
        end
    end

    fprintf('Done! Brightened images saved to: %s\n', outputFolder);
end
```

**Zaženite:**
```matlab
brighten_images
```

**OPOMBA:** To je samo "popravek" - ne bo perfektno kot pravo trenirane barvne slike!

---

## 🎯 PRIPOROČILO:

**NAJBOLJE: Kombinirajte vse 3 rešitve:**

1. **Povečajte epoch na 600** (več treninga)
2. **Različne learning rates** (G=0.0003, D=0.0001)
3. **Retrenirati model**

**To bo dalo NAJBOLJŠE rezultate:**
- Močne bakrene/zlate barve
- Pravilna svetlost
- Generator bo "confident" da outputa močne vrednosti

**Traja:** ~4-6 ur treninga (600 epoch)

**REZULTAT:** Perfektne barvne busbar slike z bakrenimi toni! 🟡✨

---

## Summary

- ✅ **Vaše učne slike:** Perfektne RGB (R=205, G=193, B=175)
- ✅ **Moj kod:** Pravilen (generator outputa RGB)
- ❌ **Problem:** Generator ni dovolj treniran (samo 300 epoch)
- ✅ **Rešitev:** Več treninga (600 epoch) + balance learning rates

**Generator SE UČI barve, ampak 300 epoch NI DOVOLJ!**

Retreniranje s 600 epoch bo dalo **močne barvne slike**! 🚀
