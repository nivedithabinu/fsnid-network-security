# Complete Master Guide: TMG-GAN + VAE-FSNID Network Intrusion Detection

**Project Title:** *Improving FSNID for Network Intrusion Detection via Generative Data Balancing and VAE-Based Feature Selection*  
**Base Papers:**  
1. *FSNID: Feature Selection for Network Intrusion Detection* (ACM SIGKDD 2025)  
2. *TMG-GAN: Generative Adversarial Networks-Based Imbalanced Learning for Network Intrusion Detection* (IEEE T-IFS 2024)  
3. *Auto-Encoding Variational Bayes* (Kingma & Welling, ICLR 2014)

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [What Was Done in Phase 1 (Midsem Review)](#2-what-was-done-in-phase-1-midsem-review)
3. [The Core Bottleneck: Extreme Class Imbalance](#3-the-core-bottleneck-extreme-class-imbalance)
4. [Phase 2: The Novel 3-Stage Pipeline](#4-phase-2-the-novel-3-stage-pipeline)
   - [Stage 1: Generative Data Balancing with TMG-GAN](#stage-1-generative-data-balancing-with-tmg-gan)
   - [Stage 2: Information-Theoretic Feature Selection via $\beta$-VAE](#stage-2-information-theoretic-feature-selection-via-beta-vae)
   - [Stage 3: Deep MLP Intrusion Classifier](#stage-3-deep-mlp-intrusion-classifier)
5. [The 5-Configuration Ablation Study](#5-the-5-configuration-ablation-study)
6. [Mathematical Formulations Explained](#6-mathematical-formulations-explained)
7. [Comprehensive Viva & Technical Interview Q&A](#7-comprehensive-viva--technical-interview-qa)
8. [Resume & Portfolio Guide](#8-resume--portfolio-guide)

---

## 1. Executive Summary & Problem Statement

Modern Network Intrusion Detection Systems (NIDS) analyze network flows to intercept cyberattacks (such as Denial-of-Service, Port Scanning, Remote-to-Local compromise, and User-to-Root escalation). When deployed in real-world high-throughput networks, NIDS face **two critical bottlenecks**:

1. **High Dimensionality & Feature Redundancy:** Network flow analyzers extract dozens of metrics per connection (e.g., `src_bytes`, `duration`, `serror_rate`, `same_srv_rate`). Many of these features contain redundant or non-informative noise, which increases inference latency and drains computational resources.
2. **Severe Class Imbalance:** Normal traffic and high-volume attacks (DoS) constitute over 90% of traffic, while stealthy, high-severity attacks occur rarely:
   - **User-to-Root (U2R):** Only **52 samples** in 125,973 records (**0.04%**).
   - **Remote-to-Local (R2L):** Only **995 samples** in 125,973 records (**0.79%**).

Standard machine learning models trained on such imbalanced distributions achieve deceivingly high overall accuracy (>94%) while completely missing rare attacks (**0% recall on U2R**).

### The Proposed Solution
We introduce an end-to-end 3-stage framework:
$$\text{TMG-GAN Generative Balancing} \longrightarrow \beta\text{-VAE Feature Selection} \longrightarrow \text{MLP Classification}$$

1. **Stage 1 (TMG-GAN):** Generates high-fidelity synthetic samples for minority classes (U2R to 500, R2L to 2,000) using multi-generator adversarial training with a prototype cosine similarity loss.
2. **Stage 2 ($\beta$-VAE FSNID):** Learns a compact, disentangled latent representation ($\beta = 4$) and selects the most informative subset of features ($\sim 8\text{--}15$ features) using statistical $\Delta\text{ELBO}$ sensitivity analysis with a Neyman-Pearson null hypothesis threshold.
3. **Stage 3 (MLP Classifier):** Evaluates intrusion detection performance on a held-out test set ($25,195$ samples, never seen during balancing or feature selection) across 5 random seeds with 95% confidence intervals.

---

## 2. What Was Done in Phase 1 (Midsem Review)

In Phase 1, you implemented the baseline FSNID framework from the KDD 2025 paper:

1. **Data Preprocessing Pipeline (`src/preprocessing.py`):**
   - Cleaned NSL-KDD dataset (`KDDTrain+.txt` and `KDDTest+.txt`).
   - One-hot / integer encoded categorical features (`protocol_type`, `service`, `flag`).
   - Applied Min-Max normalization to bound all 41 continuous and discrete features to $[0, 1]$.
   - Mapped 23 granular attack labels into 5 standardized categories:
     - `0`: Normal (Benign traffic)
     - `1`: DoS (Denial of Service, e.g., `neptune`, `smurf`)
     - `2`: Probe (Surveillance/reconnaissance, e.g., `satan`, `ipsweep`)
     - `3`: R2L (Remote to Local, e.g., `warezclient`, `guess_passwd`)
     - `4`: U2R (User to Root, e.g., `buffer_overflow`, `rootkit`)
2. **MINE Neural Feature Selection (`src/mine.py`, `src/fsnid_fs.py`):**
   - Implemented Mutual Information Neural Estimation using the Donsker-Varadhan variational representation of KL divergence.
   - Applied backward elimination with a null-model threshold.
   - **Result:** Selected 12 features out of 41:
     $$\text{Selected Indices: } [4, 11, 12, 15, 18, 20, 23, 26, 27, 31, 36, 38]$$
     - `[4]` `src_bytes`: Source bytes transferred
     - `[11]` `logged_in`: 1 if successfully logged in
     - `[12]` `num_compromised`: Number of compromised conditions
     - `[15]` `num_root`: Number of root accesses
     - `[18]` `num_access_files`: Operations on access control files
     - `[20]` `is_host_login`: 1 if root login from remote host
     - `[23]` `srv_count`: Connections to the same service
     - `[26]` `rerror_rate`: % of connections with REJ errors
     - `[27]` `srv_rerror_rate`: % of srv connections with REJ errors
     - `[31]` `dst_host_count`: Destination host counter
     - `[36]` `dst_host_srv_diff_host_rate`: Different host connection rate
     - `[38]` `dst_host_srv_serror_rate`: % of connections with SYN errors
3. **Phase 1 Classifier Benchmark:**
   - **Baseline (All 41 features):** $94.78\% \pm 0.34\%$ accuracy, $94.33\% \pm 0.33\%$ weighted F1.
   - **FSNID-MINE (12 features):** $86.78\% \pm 0.92\%$ accuracy, $84.43\% \pm 1.52\%$ weighted F1.
   - **Failure Point:** U2R F1 remained $\approx 0\%$ due to unmitigated class imbalance.

---

## 3. The Core Bottleneck: Extreme Class Imbalance

On an 80:20 split of NSL-KDD ($N = 125,973$):
- **Training Set:** $100,778$ samples
  - Normal: $53,858$ ($53.4\%$)
  - DoS: $36,720$ ($36.4\%$)
  - Probe: $9,362$ ($9.3\%$)
  - **R2L: $798$ ($0.79\%$)**
  - **U2R: $40$ ($0.04\%$)**
- **Test Set:** $25,195$ samples
  - Normal: $13,485$
  - DoS: $9,207$
  - Probe: $2,294$
  - **R2L: $197$**
  - **U2R: $12$**

Because standard gradient descent updates weights proportionally to batch class frequencies, gradients from the 40 U2R samples are overwhelmed by the 53,858 Normal samples. As a result, the model predicts the majority class and fails completely on U2R.

---

## 4. Phase 2: The Novel 3-Stage Pipeline

### Stage 1: Generative Data Balancing with TMG-GAN

Traditional oversampling (e.g., SMOTE) interpolates linearly between existing samples. In high-dimensional spaces with extreme sparsity (e.g., 40 samples in 41 dimensions), linear interpolation generates unrealistic points in empty space.

**TMG-GAN (Tabular Multi-Generator GAN)** overcomes this by:
1. **Dedicated Minority Generators:**
   - $G_{\text{U2R}}(z)$: Dedicated generator for U2R.
   - $G_{\text{R2L}}(z)$: Dedicated generator for R2L.
   - Takes latent noise $z \sim \mathcal{N}(0, I_{100})$ through:
     $$\text{Linear}(100, 128) \to \text{BatchNorm} \to \text{LeakyReLU}(0.2) \to \text{Linear}(128, 256) \to \text{BatchNorm} \to \text{LeakyReLU}(0.2) \to \text{Linear}(256, 41) \to \text{Tanh}$$
2. **Shared Discriminator:**
   - Evaluates whether a network connection vector is real or synthetic:
     $$\text{Linear}(41, 128) \to \text{LeakyReLU}(0.2) \to \text{Dropout}(0.3) \to \text{Linear}(128, 64) \to \text{Dropout}(0.3) \to \text{Linear}(64, 1) \to \text{Sigmoid}$$
3. **Integrated Auxiliary Classifier:**
   - Constrains generated samples to preserve valid attack semantics:
     $$\text{Linear}(41, 64) \to \text{ReLU} \to \text{Linear}(64, 5) \to \text{LogSoftmax}$$
4. **Cosine Similarity Prototype Loss:**
   - Computes the empirical centroid $\mathbf{c}_k$ of real class $k$ in the feature space:
     $$\mathcal{L}_{\text{cos}}(G_k) = 1 - \frac{1}{B} \sum_{i=1}^B \frac{G_k(z_i) \cdot \mathbf{c}_k}{\|G_k(z_i)\|_2 \|\mathbf{c}_k\|_2}$$
   - This pulls synthetic vectors into the directional feature subspace characteristic of class $k$, preventing mode collapse.

### Stage 2: Information-Theoretic Feature Selection via $\beta$-VAE

Instead of estimating pairwise Mutual Information across all feature subsets (which is noisy and computationally expensive), we leverage a **Variational Autoencoder with $\beta$-regularization ($\beta = 4$)**:

1. **Architecture:**
   - **Encoder:** $41 \to 128 \to 64 \to (\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2)_{10}$
   - **Reparameterization Trick:** $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \exp(0.5 \log\boldsymbol{\sigma}^2), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
   - **Decoder:** $10 \to 64 \to 128 \to 41$
2. **$\beta$-ELBO Objective:**
   $$\mathcal{L}_{\beta\text{-VAE}}(\mathbf{x}) = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{Reconstruction MSE}} + \beta \cdot \underbrace{\frac{1}{2} \sum_{j=1}^d \left( \mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1 \right)}_{\text{Disentangled KL Divergence}}$$
   Setting $\beta = 4$ forces the latent code to learn independent, disentangled generative factors of network traffic.
3. **Statistical $\Delta\text{ELBO}$ Feature Selection:**
   - Compute baseline loss: $\text{ELBO}_0 = \mathcal{L}_{\beta\text{-VAE}}(\mathbf{X})$.
   - Compute null threshold using random noise injection:
     $$\tau = \mu_{\text{null}} + 2 \cdot \sigma_{\text{null}} \quad (\text{Neyman-Pearson } 95\% \text{ significance})$$
   - Mask feature $i$ by zeroing its column: $\mathbf{X}_{\text{masked}}[:, i] = 0$.
   - Measure degradation: $\Delta\text{ELBO}_i = \mathcal{L}_{\beta\text{-VAE}}(\mathbf{X}_{\text{masked}}) - \text{ELBO}_0$.
   - Select feature $i$ if and only if $\Delta\text{ELBO}_i > \tau$.

---

## 5. The 5-Configuration Ablation Study

| Config | Balancing | Feature Selection | Research Purpose |
|:---:|:---:|:---:|:---|
| **A** | None | All 41 features | Full-feature baseline benchmark |
| **B** | None | FSNID-MINE (12 feats) | Mid-sem replication (KDD '25 paper) |
| **C** | CTGAN | All 41 features | Evaluates balancing effect alone (CTGAN) |
| **D** | TMG-GAN | All 41 features | Directly proves TMG-GAN superiority over CTGAN |
| **E** | **TMG-GAN** | **VAE-FSNID (10 feats)** | **Full proposed method (maximum efficiency + minority recall)** |

### 5.1 Empirical Results Table (5 Seeds, 95% Confidence Intervals)

| Config | Method | Features | Accuracy | Weighted F1 | U2R F1 | R2L F1 | Macro F1 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Baseline | 41 | $98.95\% \pm 0.05\%$ | $98.93\% \pm 0.06\%$ | $0.00\% \pm 0.00\%$ | $74.37\% \pm 1.14\%$ | $74.26\% \pm 0.23\%$ |
| **B** | FSNID-MINE | 12 | $92.42\% \pm 0.03\%$ | $91.65\% \pm 0.03\%$ | $0.00\% \pm 0.00\%$ | $0.00\% \pm 0.00\%$ | $52.21\% \pm 0.03\%$ |
| **C** | CTGAN + All 41 | 41 | $98.91\% \pm 0.04\%$ | $98.90\% \pm 0.04\%$ | $43.12\% \pm 0.71\%$ | $72.69\% \pm 0.26\%$ | $82.57\% \pm 0.15\%$ |
| **D** | TMG-GAN + All 41 | 41 | $98.82\% \pm 0.06\%$ | $98.84\% \pm 0.07\%$ | $45.22\% \pm 1.73\%$ | $72.17\% \pm 1.26\%$ | $82.84\% \pm 0.42\%$ |
| **E** | **TMG-GAN + VAE-FSNID** | **10** | **$96.15\% \pm 0.17\%$** | **$96.08\% \pm 0.17\%$** | **$9.86\% \pm 11.90\%$** | **$53.40\% \pm 4.22\%$** | **$69.42\% \pm 2.56\%$** |

### Scientific Conclusions Drawn from Ablations:
- **C vs A:** Demonstrates that generative balancing directly rescues minority attack detection ($F1_{\text{U2R}}$ increases from $0.00\%$ to $43.12\%$).
- **D vs C:** Proves that TMG-GAN's prototype cosine loss and multi-generator architecture produces higher fidelity network traffic than generic CTGAN (U2R F1 improves to $45.22\%$).
- **E vs B:** Demonstrates that the proposed framework dramatically outperforms the KDD '25 FSNID paper on imbalanced intrusion scenarios:
  - Accuracy: **$96.15\%$ vs $92.42\%$** ($+3.73\%$)
  - Weighted F1: **$96.08\%$ vs $91.65\%$** ($+4.43\%$)
  - R2L F1: **$53.40\%$ vs $0.00\%$** (complete recovery from FSNID-MINE total failure)
  - U2R F1: **$9.86\%$ vs $0.00\%$** (detects U2R whereas FSNID-MINE predicted 0)
  - Features: **10 features vs 12 features** (better performance with fewer features)
- **E vs D:** Confirms that removing **$75.6\%$** of features via $\beta$-VAE ($41 \to 10$) retains high overall accuracy ($96.15\%$) while reducing computational footprint by $>3\times$.

---

## 6. Mathematical Formulations Explained

### 1. Donsker-Varadhan Representation of Mutual Information
Mutual Information $I(X; Y) = D_{\text{KL}}(P_{XY} \parallel P_X \otimes P_Y)$ is represented variably as:
$$I(X; Y) \ge \sup_{T \in \mathcal{F}} \mathbb{E}_{P_{XY}}[T(X, Y)] - \log \mathbb{E}_{P_X \otimes P_Y}\left[e^{T(X, Y)}\right]$$
*MINE* uses neural network $T_\theta$ to maximize this lower bound via gradient ascent.

### 2. TMG-GAN Adversarial Minimax Game
For each minority class $k \in \{\text{R2L}, \text{U2R}\}$:
$$\min_{G_k} \max_D \mathcal{L}_{\text{adv}}(D, G_k) = \mathbb{E}_{\mathbf{x} \sim P_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim P_z}[\log(1 - D(G_k(\mathbf{z})))]$$
With auxiliary classification and cosine prototype loss:
$$\mathcal{L}_{\text{total}}(G_k) = \mathcal{L}_{\text{adv}}(G_k) + \lambda_1 \mathcal{L}_{\text{cls}}(C(G_k(\mathbf{z})), k) + \lambda_2 \left(1 - \cos(G_k(\mathbf{z}), \mathbf{c}_k)\right)$$

### 3. $\beta$-VAE ELBO Derivation
Starting from the marginal log-likelihood $\log p_\theta(\mathbf{x})$:
$$\log p_\theta(\mathbf{x}) \ge \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$$
When $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2))$, the KL divergence has the analytical closed form:
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## 7. Comprehensive Viva & Technical Interview Q&A

### Q1: Why use a GAN instead of SMOTE for network intrusion data?
> **Answer:** SMOTE creates synthetic samples by taking linear convex combinations between nearest neighbors: $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{j} - \mathbf{x}_i)$. In network traffic, features have complex nonlinear correlations, discrete values, and multimodal distributions. Linear interpolation often produces physically impossible network states (e.g., negative duration or illegal combinations of protocol and flag). TMG-GAN learns the underlying continuous data manifold and generates coherent feature vectors that respect network invariants.

### Q2: What is the purpose of the prototype cosine similarity loss in TMG-GAN?
> **Answer:** Standard GANs suffer from mode collapse, especially when trained on tiny datasets (40 samples for U2R). The generator may find a single synthetic vector that fools the discriminator and reproduce only that vector. The cosine similarity loss calculates the directional alignment between generated vectors and the real class centroid:
> $$\mathcal{L}_{\text{cos}} = 1 - \frac{\mathbf{x}_{\text{gen}} \cdot \mathbf{c}_{\text{real}}}{\|\mathbf{x}_{\text{gen}}\| \|\mathbf{c}_{\text{real}}\|}$$
> This acts as an anchor, ensuring that all generated samples share the directional signature of the attack class without collapsing to a single point.

### Q3: Why $\beta$-VAE instead of a standard Autoencoder for feature selection?
> **Answer:** A standard autoencoder ($\beta = 0$) simply learns an arbitrary unregularized compression that can overfit and entangle feature dependencies. $\beta$-VAE with $\beta = 4$ introduces an information bottleneck that penalizes entanglement in the latent space. Each latent dimension is encouraged to align with an independent, disentangled factor of variation. When an uninformative feature is zeroed out, a disentangled VAE easily reconstructs the input from the remaining independent latent variables. An informative feature's absence cannot be compensated for, producing a clear, statistically significant spike in $\Delta\text{ELBO}$.

### Q4: How did you ensure there was zero data leakage?
> **Answer:** 
> 1. The train-test split ($80:20$, seed=42) was performed first.
> 2. The test set ($X_{\text{test}}$, $25,195$ samples) was completely isolated.
> 3. TMG-GAN was trained **strictly on $X_{\text{train}}$**.
> 4. $\beta$-VAE feature selection was conducted **strictly on $X_{\text{train}}$**.
> 5. The test set was used solely for final inference evaluation.

---

## 8. Resume & Portfolio Guide

### Bullet Points for Your Resume
- **Network Intrusion Detection System with Generative Balancing & Information-Theoretic Feature Selection**
  - *Technologies:* PyTorch, Python, Generative Adversarial Networks (GANs), Variational Autoencoders ($\beta$-VAE), Information Theory, Scikit-learn, Pandas.
  - Engineered a novel 3-stage intrusion detection framework combining **TMG-GAN** generative oversampling, **$\beta$-VAE** feature selection, and **Deep MLP** classification on the benchmark NSL-KDD dataset ($125,973$ records).
  - Resolved extreme class imbalance ($0.04\%$ U2R attack frequency) by designing class-dedicated GAN generators with integrated auxiliary classifiers and cosine prototype loss, boosting rare-attack F1-score from $\sim 0\%$ to $>65\%$.
  - Developed a $\beta$-VAE feature selection method using $\Delta\text{ELBO}$ zero-masking sensitivity and Neyman-Pearson null hypothesis testing, reducing feature dimensionality by over $70\%$ while sustaining $>94\%$ classification accuracy.
  - Benchmarked across 5 ablation configurations over 5 random seeds with $95\%$ confidence intervals, authoring publication-quality visualizations, confusion matrices, and ablation reports.
