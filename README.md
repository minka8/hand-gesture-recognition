# Rock–Paper–Scissors Hand Gesture Recognition (Hypothesis Testing)

This repository contains a Jupyter notebook (`hand_gesture_recoginition.ipynb`) that implements a hand-gesture recognition system for **Rock / Paper / Scissors** based on **feature extraction from binary masks** and **Gaussian (MAP) multi-hypothesis classification**.  
The notebook also includes an additional **2-class parametric classification** (linear + quadratic classifier) for selected classes/features.

---

## Dataset Assumption

The preprocessing is built for images where the hand is recorded on a **green background** (chroma key).
Green pixels are removed by HSV thresholding.

The notebook expects three folders (paths are currently set locally and must be updated):
- `rock/`
- `paper/`
- `scissors/`

Each folder contains images of the corresponding gesture.

---

## Pipeline

### 1) Preprocessing

The preprocessing step is implemented in:

- `imgBin(img)`  
  - Converts RGB → HSV  
  - Thresholds **green color range** using `cv2.inRange`  
  - Inverts the mask (hand becomes foreground)
  - Applies median filtering
  - Performs a fixed threshold relative to max intensity to obtain a binary image

- `filterImg(img)`  
  - Median filter (`ndimage.median_filter`, size=3)

- `removeEdge(img)`  
  - Crops the binary image to a bounding box by scanning borders until a minimum percentage of nonzero pixels is found

- `preprocessing(img)`  
  - Runs: `imgBin → filterImg → removeEdge`

Output of preprocessing: **binary image** where foreground pixels are `255`.

---

### 2) Feature Extraction

Features are computed in `extractFeatures(img)` and returned as a 5D vector:

| Index | Feature |
|------:|---------|
| f0 | Area = number of nonzero pixels |
| f1 | Perimeter (skimage `perimeter(img == 255, neighborhood=8)`) |
| f2 | Area / Perimeter |
| f3 | Compactness = (4πA) / P² (if P>0 else 0) |
| f4 | Left/Right foreground ratio (count of 255 pixels in left half divided by right half) |

For a class folder, features are extracted using:
- `patternFeatures(folder, n_img)`

---

### 3) Feature Scaling

The notebook performs min–max scaling to `[0,1]` using:
- `scaling(feature)`

Then scaled features are transposed and merged into one dataset with labels:
- 0 = paper
- 1 = rock
- 2 = scissors

---

### 4) Train/Test Split

`train_test_split` is used with:
- `test_size=0.3`
- `stratify=y`
- `random_state=42`

---

## Multi-class Classification: Gaussian MAP (3 hypotheses)

The notebook estimates per-class parameters from the training set:

- Mean vectors: `M1, M2, M3`
- Covariance matrices: `S1, S2, S3`
- Priors: `P1, P2, P3` (computed from class sample counts)

Gaussian PDF is implemented in:
- `gaussian_pdf(X, M, S)`

Classification rule (MAP):
- `classifyMul(X)` computes `gk = Pk * gaussian_pdf(x, Mk, Sk)` and picks `argmax`.

---

## Evaluation

The notebook evaluates predictions on the test set using:

- `confusion_matrix`
- `accuracy_score`
- `precision_score (macro)`
- `recall_score (macro)`
- `f1_score (macro)`
- `balanced_accuracy_score`

The confusion matrix is plotted as a heatmap.

---

## Feature Separability Analysis (2 classes)

The notebook includes helper plots:
- `plotFeatureHist(...)` – histogram of a chosen feature for two classes
- `plotClassAndFeatures(...)` – scatter plot for two chosen features

In the notebook, for **Paper (0)** vs **Rock (1)**, the pair `(f0, f4)` is marked as the best separation (`#najbolje`).

---

## Additional 2-class Parametric Classifiers (Paper vs Rock)

### Linear classifier (desired outputs)
Implemented with:
- `desiredOutput(K1, K2, Gamma)`

A linear decision function `g(x)` is applied and a 2-class confusion matrix is computed.

### Quadratic classifier (desired outputs)
Implemented with:
- `quad_features(X)` (builds quadratic feature map)
- `desiredOutputQuadratic(K1, K2, Gamma)` (pseudo-inverse solution)

---

