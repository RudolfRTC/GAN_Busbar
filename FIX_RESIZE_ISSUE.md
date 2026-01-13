# REŠITEV ZA RESIZE PROBLEM

## Problem
Vaše slike so **256x256** (kvadratne), ampak kod jih resize-a na **64x128** (pravokotne).

**PROBLEM:**
- ❌ Spremeni aspect ratio (kvadrat → pravokotnik)
- ❌ Slike so raztegnjene/stisnjene
- ❌ Izguba kvalitete (256 → 64 višina)

---

## ✅ REŠITEV: Uporabi kvadratne dimenzije

Za vaše 256x256 busbar slike uporabite **trainSize = [128 128]** (kvadratno)!

---

## KORAK 1: Spremeni train_gan.m

```matlab
% V train_gan.m vrstica 33, spremeni:
% STARO:
params.trainSize = [64 128];

% NOVO:
params.trainSize = [128 128];  % Kvadratno za vaše 256x256 slike!
```

---

## KORAK 2: Uporabi generator za kvadratne slike

Generator trenutno podpira samo [64 128]. Morate uporabiti **buildGenerator_square.m**.

Zaženite tale script da ga kreate:

```matlab
% V MATLAB:
create_square_generator
```

Ta bo kreiral `buildGenerator_square.m` ki podpira [128 128].

---

## KORAK 3: Spremeni train_gan.m da uporablja kvadratni generator

```matlab
% V train_gan.m vrstica ~95, spremeni:
% STARO:
netG = buildGenerator(params);

% NOVO:
netG = buildGenerator_square(params);
```

---

## KORAK 4: Retrenirati

```matlab
clear all
train_gan  % Bo sedaj generiral 128x128 kvadratne slike!
```

---

## ALTERNATIVA: Uporabi 64x64 (hitrejše)

Če želite še hitrejši trening:

```matlab
params.trainSize = [64 64];  % Manjše kvadratne slike
```

To bo:
- ✅ Hitrejši trening (~2x)
- ✅ Manj VRAM
- ❌ Nekoliko slabša kvaliteta

---

## Summary

| Dimenzija | Aspect Ratio | Kvaliteta | Hitrost | VRAM |
|-----------|--------------|-----------|---------|------|
| [64 128] | ❌ 2:1 (raztegnjeno) | Nizka | Hitra | Nizka |
| [64 64] | ✅ 1:1 (kvadrat) | Srednja | Hitra | Nizka |
| **[128 128]** | ✅ **1:1 (PRIPOROČENO)** | **Visoka** | Srednja | Srednja |
| [256 256] | ✅ 1:1 (najboljše) | Najvišja | Počasna | Visoka |

**PRIPOROČILO: Uporabite [128 128]!**

---

## OPOMBA

Generator bo mogoče potreboval **več epoch** (npr. 500) za višjo resolucijo [128 128]!
