# MTUNet++ — Breast Cancer Detection in Ultrasound Images

A deep learning project that looks at breast ultrasound images and does two things at once — figures out what the anomaly is (normal / benign / malignant) and draws a mask around it. One model, one pass, both outputs.

---

## Why I Built This

Ultrasound-based breast cancer screening is one of those problems where the bottleneck isn't data collection, it's consistent interpretation. I wanted to see if a shared-encoder architecture could learn better representations by solving classification and segmentation jointly rather than treating them as separate problems.

The other thing I experimented with: instead of just trusting the neural network's final softmax output for classification, I tapped into the internal 512-dim feature vector midway through the network and trained an SVM on that. Then blended both predictions. Ended up being more stable and hit 81.6% accuracy on the curated dataset.

---

## Dataset

**BUSI (Breast Ultrasound Images)** — publicly available, 780 images. After removing near-duplicates using SSIM similarity, I worked with a cleaner set of 450 images at 128×128.

| Class | Images |
|-------|--------|
| Normal | 64 |
| Benign | 222 |
| Malignant | 164 |

> Dataset not included. Download BUSI [here](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) and run `src/dataset/Curated_BUSI_preprocessing.py` to generate the curated version.

---

## How It Works

The UNet++ backbone acts as a shared encoder. One decoder branch produces segmentation masks. The classification branch pools the encoder output down to a 512-dim vector — I hooked into that layer to extract features and run an SVM alongside the neural network. Final prediction blends both.
```
Shared UNet++ Encoder
│
├──► Segmentation Decoder → binary mask
│
└──► Classification Branch
→ 512-dim bottleneck features  ← SVM trained here
→ Linear layers → softmax
Final output = weighted blend of NN + SVM predictions
```
---

## Project Structure
```
MTUNet++/
├── src/
│   ├── models/
│   │   ├── multitask/         # shared encoder + task heads
│   │   ├── classification/
│   │   └── segmentation/
│   ├── dataset/
│   │   ├── BUSI_dataset.py
│   │   └── BUSI_dataloader.py
│   └── utils/
│       ├── metrics.py         # Dice, Jaccard, Hausdorff
│       ├── criterions.py
│       ├── visualization.py
│       └── miscellany.py
├── training_multitask.py      ← start here
├── config.yaml
├── requirements.txt
└── README.md
```
---

## Running It

```bash
git clone https://github.com/anushkakuchimanchi/breast-cancer-detection-mtunet.git
cd breast-cancer-detection-mtunet
pip install -r requirements.txt

# prep dataset
python src/dataset/Curated_BUSI_preprocessing.py

# train
python training_multitask.py
```

Paths, batch size, and ensemble blend weights are all in `config.yaml`.

---

## Stack

Python 3.11 · PyTorch 2.5.1 · MONAI · scikit-learn · OpenCV · NumPy · pandas · matplotlib

---

## Credits

The UNet++ multitask backbone builds on open-source work by Aumente-Maestro et al. (2024), available at [caumente/multi_task_breast_cancer](https://github.com/caumente/multi_task_breast_cancer). Licensed under Apache 2.0 — see `LICENSE`.
