# GAN za Generiranje Sintetičnih Slik Industrijskih Kosov

Ta repozitorij vsebuje **dve implementaciji** GAN-ov za generiranje sintetičnih slik industrijskih kosov iz majhnega dataseta (~70 slik):

## 📂 Implementaciji

### 🔵 MATLAB DCGAN (ta branch: `main`)
MATLAB implementacija DCGAN (Deep Convolutional GAN) za generiranje sintetičnih slik. Sistem je optimiziran za produktne fotografije na belem ozadju in vključuje avtomatsko obrezovanje, močno augmentacijo ter stabilizacijske tehnike za trening z malo podatki.

**Primerno za:**
- Uporabnike MATLAB-a
- Hitro prototipiranje
- Izobraževalne namene

### 🐍 Python StyleGAN2-ADA (branch: `python-stylegan2-ada`)
Python implementacija z **NVIDIA StyleGAN2-ADA** - state-of-the-art GAN arhitektura z adaptivno augmentacijo, optimizirana za majhne datasete.

**Prednosti:**
- ⭐ **Občutno boljša kvaliteta slik** kot DCGAN
- 🔄 **Adaptivna augmentacija (ADA)** preprečuje overfitting
- 🎯 **Optimizirano za malo podatkov** (~50-100 slik)
- 🚀 **Production-ready** pipeline

**Za preklop na Python implementacijo:**
```bash
git checkout python-stylegan2-ada
```

**Podrobna navodila:** Glej `python_stylegan2ada/README.md` v Python branchu.

---

## MATLAB DCGAN Dokumentacija

Spodaj so navodila za MATLAB implementacijo. Za Python StyleGAN2-ADA glej branch `python-stylegan2-ada`.

## Zahteve

- MATLAB R2020b ali novejši
- Deep Learning Toolbox
- GPU (priporočeno, vendar ni obvezno - koda avtomatsko zazna in uporabi GPU, če je na voljo)

## Hitri Začetek

### 1. Priprava Podatkov

Ustvarite mapo z vašimi slikami:
```
./data/images/
```

Podprti formati: `.jpg`, `.jpeg`, `.png`, `.bmp`

**Pomembno:** Slike naj bodo produktne fotografije na belem ali svetlem ozadju (optimalno za avtomatsko obrezovanje).

### 2. Zagon Treninga

Odprite MATLAB in zaženite:
```matlab
train_gan
```

To bo:
1. Prebralo vaše slike
2. Avtomatsko zaznalo RGB vs grayscale
3. Naredilo predobdelavo (auto-crop, resize, augmentacija)
4. Treniralo GAN za ~300 epoch
5. Generiralo 2000 sintetičnih slik

### 3. Rezultati

Po treningu boste našli:
- **Preview slike med treningom:** `./outputs/preview/`
- **Trenirani modeli:** `./outputs/models/` (generator.mat, discriminator.mat)
- **Sintetične slike:** `./outputs/synthetic/` (synthetic_000001.png do synthetic_002000.png)

## Struktura Projekta

```
GAN_Busbar/
├── train_gan.m                      # Glavna skripta za trening
├── buildGenerator.m                 # DCGAN generator arhitektura
├── buildDiscriminator.m             # DCGAN discriminator arhitektura
├── preprocessAndLoadDatastore.m     # Nalaganje in predobdelava slik
├── saveImageGrid.m                  # Shranjevanje preview gridov
├── generateSynthetic.m              # Generiranje sintetičnih slik
├── data/
│   └── images/                      # Vaše vhodne slike (jpg/png)
└── outputs/
    ├── preview/                     # Preview grid med treningom
    ├── models/                      # Trenirani modeli
    └── synthetic/                   # Generirane sintetične slike
```

## Parametri (Prilagoditev)

Odprite `train_gan.m` in poiščite sekcijo `PARAMETERS`. Tukaj lahko spreminjate:

### Osnovni Parametri
```matlab
params.imageSize = 128;              % Velikost slik (64 ali 128)
params.latentDim = 100;              % Dimenzija latentnega vektorja
params.numEpochs = 300;              % Število epoch (300-500 za malo podatkov)
params.miniBatchSize = 16;           % Batch size (8, 16, 32)
params.learnRate = 0.0002;           % Learning rate
params.numSynthetic = 2000;          % Število sintetičnih slik (2000-5000)
```

### Auto-Crop Parametri
```matlab
params.autoCrop = true;              % Vklopi/izklopi auto-crop
params.cropThreshold = 0.85;         % Prag za belo ozadje (0-1)
                                     % Nižja vrednost = strožje obrezovanje
                                     % Višja vrednost = ohrani več ozadja
```

### Stabilizacija (za malo podatkov)
```matlab
params.labelSmoothing = 0.9;         % Real labels = 0.9 namesto 1.0
params.instanceNoise = 0.05;         % Instance noise za stabilnost
params.noiseDecay = 0.995;           % Decay rate za instance noise
```

## Funkcionalnosti

### 1. Avtomatska Detekcija RGB/Grayscale
Koda avtomatsko zazna, ali so vaše slike v barvi ali sivi skali, in ustrezno prilagodi trening.

### 2. Avtomatsko Obrezovanje (Auto-Crop)
- Zazna objekt na svetlem ozadju
- Izračuna bounding box okoli največjega objekta
- Doda 10% padding
- Optimizira za produktne fotografije

Če auto-crop ne deluje dobro (npr. ozadje ni dovolj svetlo), nastavite:
```matlab
params.autoCrop = false;
```

### 3. Močna Data Augmentation
Za stabilnost pri malo podatkih se uporablja:
- Random horizontal flip
- Majhna rotacija (±10°)
- Translation (premik slike)
- Brightness/contrast jitter
- Gaussian noise

### 4. Stabilizacijske Tehnike
- **Label Smoothing:** Real labels = 0.9 namesto 1.0
- **Instance Noise:** Dodajanje šuma na vhodne slike discriminatorja
- **Dropout (30%):** V discriminatorju za regularizacijo
- **Batch Normalization:** Za stabilnost treninga

### 5. GPU Podpora
Koda avtomatsko zazna GPU in ga uporabi, če je na voljo:
```matlab
params.executionEnvironment = 'auto'; % 'auto', 'gpu', ali 'cpu'
```

## TROUBLESHOOTING

### Problem 1: Mode Collapse (vse generirane slike so enake)

**Simptomi:**
- Preview slike so vse zelo podobne ali identične
- Generator loss strmo pade na zelo nizko vrednost

**Rešitve:**
1. **Zmanjšaj learning rate:**
   ```matlab
   params.learnRate = 0.0001;  % Namesto 0.0002
   ```

2. **Povečaj instance noise:**
   ```matlab
   params.instanceNoise = 0.1;  % Namesto 0.05
   ```

3. **Povečaj label smoothing:**
   ```matlab
   params.labelSmoothing = 0.85;  % Namesto 0.9
   ```

4. **Zmanjšaj batch size:**
   ```matlab
   params.miniBatchSize = 8;  % Namesto 16
   ```

5. **Treniraj dlje:**
   ```matlab
   params.numEpochs = 500;  % Namesto 300
   ```

### Problem 2: Previewji so sami beli/prazni

**Simptomi:**
- Generirane slike so prazne, bele ali samo šum

**Vzroki in rešitve:**

1. **Discriminator je prezahteven:**
   - Generator ne more "prevarati" discriminatorja
   ```matlab
   % V buildDiscriminator.m zmanjšaj dropout:
   dropoutLayer(0.2, ...)  % Namesto 0.3
   ```

2. **Learning rate je previsok:**
   ```matlab
   params.learnRate = 0.0001;
   ```

3. **Premalo epoch:**
   - Za 70 slik lahko traja 300+ epoch, preden vidite dobre rezultate
   ```matlab
   params.numEpochs = 500;
   ```

4. **Preveri normalizacijo slik:**
   - Slike morajo biti v [-1, 1] range
   - Preveri v `preprocessAndLoadDatastore.m`

### Problem 3: GPU Memory Issue

**Simptomi:**
- Error: "Out of memory" ali "GPU memory exceeded"

**Rešitve:**

1. **Zmanjšaj batch size:**
   ```matlab
   params.miniBatchSize = 8;  % Ali celo 4
   ```

2. **Zmanjšaj image size:**
   ```matlab
   params.imageSize = 64;  % Namesto 128
   ```

3. **Uporabi CPU (počasneje, vendar deluje):**
   ```matlab
   params.executionEnvironment = 'cpu';
   ```

4. **Počisti GPU memory pred treningom:**
   ```matlab
   gpuDevice(1); % Reset GPU
   ```

### Problem 4: Discriminator Loss gre v 0

**Simptomi:**
- D Loss hitro pade na ~0.00
- G Loss eksplodira ali stagnira

**Vzrok:** Discriminator je preveč močan in vedno pravilno loči real/fake.

**Rešitve:**

1. **Label smoothing:**
   ```matlab
   params.labelSmoothing = 0.8;  % Bolj agresivno smoothing
   ```

2. **Treniraj generator večkrat:**
   V `train_gan.m` v training loop dupliciraj generator update:
   ```matlab
   % Train Generator twice
   [gradG, lossG] = dlfeval(@modelGradientsG, netD, netG, Z);
   [netG, avgG, avgGS] = adamupdate(netG, gradG, avgG, avgGS, iteration, ...
       params.learnRate, params.beta1);
   [gradG, lossG] = dlfeval(@modelGradientsG, netD, netG, Z);
   [netG, avgG, avgGS] = adamupdate(netG, gradG, avgG, avgGS, iteration, ...
       params.learnRate, params.beta1);
   ```

### Problem 5: Koda javi "No images found"

**Vzrok:** Slike niso v pravilni mapi ali imajo nepodprt format.

**Rešitev:**
- Preveri, da so slike v `./data/images/`
- Preveri format: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Preveri uppercase/lowercase (Linux je case-sensitive)

### Problem 6: Slike so premajhne (< 70)

**Opozorilo:** Koda bo izpisala warning in samodejno zmanjšala batch size.

**Če imaš manj kot 30 slik:**
- GAN bo zelo težko treniral
- Priporočeno: zberi vsaj 50-100 slik
- Alternativa: uporabi transfer learning ali StyleGAN2-ADA (advanced)

### Problem 7: Auto-crop ne deluje

**Simptomi:**
- Preview slike imajo še vedno veliko belo ozadje
- Objekti niso pravilno obrezani

**Rešitve:**

1. **Prilagodi threshold:**
   ```matlab
   params.cropThreshold = 0.75;  % Nižja vrednost = strožje
   ```

2. **Izklopi auto-crop in ročno obreži slike:**
   ```matlab
   params.autoCrop = false;
   ```
   Nato uporabi ImageJ, Photoshop ali drug tool za crop.

## Napredne Možnosti

### Generiranje dodatnih sintetičnih slik

Če želite kasneje generirati še več slik, naložite shranjeni model:

```matlab
% Naloži model
load('./outputs/models/generator.mat', 'netG', 'params');

% Nastavi število novih slik
params.numSynthetic = 5000;

% Generiraj
generateSynthetic(netG, params);
```

### Fine-tuning na novih slikah

Če dobite dodatne slike in želite nadaljevati trening:

```matlab
% Naloži modele
load('./outputs/models/generator.mat', 'netG', 'params');
load('./outputs/models/discriminator.mat', 'netD');

% Dodaj nove slike v ./data/images/
% Ponovno zaženi train_gan.m
% (lahko zmanjšate numEpochs za krajši trening)
```

### Spreminjanje arhitekture

Če želite eksperimentirati z različnimi arhitekturami:
- **Generator:** Uredi `buildGenerator.m` (število filtrov, plasti)
- **Discriminator:** Uredi `buildDiscriminator.m` (dropout rate, plasti)

**Opomba:** To zahteva poznavanje GAN arhitektur in MATLAB Deep Learning Toolbox.

## Kako Oceniti Rezultate

### Dobri znaki:
- Preview slike postopoma izboljšujejo kvaliteto
- D Loss in G Loss sta relativno stabilna (ne eksplodirajo)
- Generirane slike imajo jasen objekt (ne samo šum)
- Varijacija med generiranimi slikami (ne vse enake)

### Slabi znaki:
- Mode collapse: vse slike enake
- Generirane slike so samo šum ali bele
- Loss vrednosti eksplodirajo (NaN ali zelo visoke vrednosti)
- Discriminator loss = 0 (premočan discriminator)

### Tipična Loss vrednosti:
- D Loss: 0.3 - 1.5 (ni strogo, vendar približno)
- G Loss: 0.5 - 3.0
- Pomembno: **stabilnost**, ne absolutne vrednosti

## Uporaba Sintetičnih Slik

Ko imate generirane sintetične slike (`./outputs/synthetic/`), jih lahko uporabite za:

1. **Data Augmentation:** Združite original + sintetične slike za trening klasifikacijske mreže
2. **Testiranje:** Testirajte robustnost vašega modela
3. **Vizualizacija:** Pokažite različne variacije izdelka

**Priporočilo:** Vedno preverite kvaliteto sintetičnih slik ročno, preden jih uporabite za trening!

## Dodatne Informacije

### Zakaj DCGAN?
- Preizkušena arhitektura za generiranje slik
- Relativno stabilen trening
- Dobri rezultati tudi z majhnimi dataseti (s pravilnimi nastavitvami)

### Zakaj 300+ epoch?
Pri malo podatkih (~70 slik) GAN potrebuje več iteracij za konvergenco. Ne skrbite, če po 50 epochs slike še ne izgledajo dobro - to je normalno!

### Časovna Zahtevnost
- **Z GPU (npr. GTX 1060):** ~2-4 ure za 300 epoch (128x128)
- **Z CPU:** ~10-20 ure za 300 epoch (128x128)
- **Hitrejša možnost:** Zmanjšaj na 64x64 ali manj epoch

## Reference

- **DCGAN paper:** Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
- **MATLAB Deep Learning Toolbox:** https://www.mathworks.com/help/deeplearning/

## Kontakt & Podpora

Za dodatne težave:
1. Preverite MATLAB verzijo (`ver`)
2. Preverite Deep Learning Toolbox (`ver deeplearning`)
3. Preverite GPU podpora (`gpuDevice`)

## Licenca

Ta koda je namenjena izobraževalnim in raziskovalnim namenom.

---

**Uspešen trening! 🚀**
