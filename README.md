# TMG-GAN + VAE-FSNID: Extending FSNID for Network Intrusion Detection

> **Course**: Artificial Intelligence — Semester 05
> **Institution**: IIIT Vadodara
> **Dataset**: NSL-KDD
> **Base Paper**: Westphal et al., *FSNID: Feature Selection for Network Intrusion Detection*, KDD 2025
> **Extension Reference**: Ding et al., *TMG-GAN: GAN-Based Imbalanced Learning for Network Intrusion Detection*, IEEE T-IFS 2024

---

## Project Overview

This project replicates and extends the FSNID (Feature Selection for Network Intrusion Detection) algorithm. FSNID uses Mutual Information Neural Estimation (MINE) to select the most informative features from network traffic data for attack classification.

**Phase 1 (Mid-Semester)**: Replicated FSNID on NSL-KDD — achieved 94.78% baseline accuracy, 86.78% with 12 MINE-selected features (70.7% dimensionality reduction).

**Phase 2 (Extension)**: Addresses two limitations of the original FSNID:
1. **Class imbalance**: NSL-KDD has only 40 U2R and 798 R2L samples vs 67k Normal samples
2. **MINE instability**: MINE trains a separate network per feature, making it slow and noisy

Our solution: **TMG-GAN** (multi-generator GAN) for data balancing + **β-VAE** reconstruction sensitivity for feature selection.

---

## Project Structure

```
fsnid-network-security/
│
├── data/
│   ├── raw/
│   │   ├── KDDTrain+.txt          # NSL-KDD raw training data (125,973 samples)
│   │   └── KDDTest+.txt           # NSL-KDD raw test data
│   └── processed/
│       └── NSL-KDD/
│           ├── X_train.npy        # Preprocessed features (125,973 × 41)
│           ├── y_train.npy        # Attack labels (string)
│           ├── X_test.npy         # Preprocessed test features
│           └── y_test.npy         # Test labels
│
├── src/
│   ├── classifier.py              # MLP classifier [input→250→250→5]
│   ├── fsnid_fs.py                # FSNID feature selector (MINE-based)
│   ├── mine.py                    # MINE + mine_fa estimators
│   ├── models.py                  # Base neural network definitions
│   ├── preprocessing.py           # Data loading and preprocessing
│   ├── tmg_gan.py                 # TMG-GAN: multi-generator GAN [NEW]
│   ├── vae.py                     # β-VAE model [NEW]
│   └── vae_fs.py                  # VAE-based feature selector [NEW]
│
├── results/
│   ├── phase1_results.txt         # Phase 1 replication results
│   └── extension_results.json     # Phase 2 extension results (auto-generated)
│
├── docs/
│   └── architecture.md            # Detailed system architecture
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook
│
├── run_classifier.py              # Phase 1: FSNID replication pipeline
├── run_extension.py               # Phase 2: TMG-GAN + VAE-FSNID pipeline
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Setup

```bash
git clone https://github.com/nivedithabinu/fsnid-network-security
cd fsnid-network-security

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

---

## Running the Code

### Phase 1 — FSNID Replication
```bash
python run_classifier.py
```
Outputs: Baseline (41 features) vs FSNID-MINE (12 features) accuracy and F1.

### Phase 2 — Extension
```bash
python run_extension.py
```
Outputs: 4-config ablation study. Results saved to `results/extension_results.json`.

---

## Results Summary

### Phase 1 (run_classifier.py)
| Method | Features | Accuracy | Weighted F1 |
|---|---|---|---|
| Baseline (All) | 41 | 94.78% ± 0.34% | 94.33% ± 0.33% |
| FSNID-MINE | 12 | 86.78% ± 0.92% | 84.43% ± 1.52% |

### Phase 2 (run_extension.py)
See `results/extension_results.json` for full results after running.

---

## Classes (NSL-KDD)
| Class | Label | Train Samples |
|---|---|---|
| 0 | Normal | 53,858 |
| 1 | DoS | 36,720 |
| 2 | Probe | 9,362 |
| 3 | R2L | 798 |
| 4 | U2R | 40 |

---

## References

1. Westphal, C., Hailes, S., Musolesi, M. (2025). *Feature Selection for Network Intrusion Detection*. KDD 2025. DOI: 10.1145/3690624.3709339
2. Ding, H., Sun, Y., Huang, N., Shen, Z., Cui, X. (2024). *TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection*. IEEE Transactions on Information Forensics and Security, 19.
3. Kingma, D.P., Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR 2014.
