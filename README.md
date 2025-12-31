# Vision-Driven Entropy Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository presents a **research prototype for vision-based physical entropy acquisition** and its integration into a standard cryptographic image encryption pipeline.

The system captures motion dynamics from a real-world video source (e.g., a lava lamp), derives a time-varying *motion-activity signal* using multiple computer vision features, and conditions this signal to generate encryption keys for **AES-256-GCM**.

> **Scope Clarification**
> This work does **not** propose a new cryptographic primitive. It focuses on entropy acquisition, conditioning, and empirical validation, while relying exclusively on standard, well-studied cryptographic components (HKDF, AES-GCM).

## Key Contributions

* **Motion-Gated Entropy Pipeline**: Extracts entropy proxies using optical flow, color drift, corner density, and texture features.
* **Failure-Safe Design**: Explicit rejection of static or near-static scenes to prevent low-entropy seeding.
* **Modular Architecture**: Clean separation between computer vision (entropy source) and cryptographic logic (encryption).
* **Maximalist Validation**: Comprehensive evaluation covering entropy behavior, temporal decorrelation, sensitivity (NPCR/UACI), and downstream cryptographic effects.
* **Reproducibility**: Deterministic experiments and a modular codebase suitable for further academic or applied research.

## System Architecture

```
[Video Source]
      |
      v
(Computer Vision Engine)
      |
      |-- Dense Optical Flow
      |-- HSV Color Drift
      |-- Corner Density
      |-- Texture Entropy
      v
<Motion Gating Check> ----(Static / Noise)----> [REJECT]
      |
   (Active)
      |
      v
[Entropy Aggregation]
      |
      v
[Conditioner: SHA-256 + HKDF]
      |
      v
[AES-256 Key & IV]
      |
      v
[Plaintext Image] --(AES-GCM)--> [Ciphertext]
```

### 1. Physical Entropy Capture (Computer Vision)

Entropy proxies are computed from consecutive video frames using:

* **Dense Optical Flow (Farnebäck)**: Captures fluid and non-rigid motion.
* **HSV Color Drift**: Measures environmental and illumination changes.
* **Shi–Tomasi Corner Density**: Quantifies topological complexity.
* **LBP Texture Entropy**: Encodes fine-grained local texture variation.

**Motion Gating** suppresses output when both motion magnitude and color drift fall below a threshold, preventing static sensor noise from contributing entropy.

### 2. Entropy Aggregation

Extracted features are combined into a single scalar *motion-activity score* using fixed, interpretable weights. This value represents **physical activity**, not cryptographic entropy directly.

### 3. Entropy Conditioning

The activity signal is:

1. Serialized into a byte stream.
2. Mixed with the SHA-256 hash of the target image.
3. Expanded using **HKDF-SHA256**.

This ensures the final seed depends jointly on the physical entropy source and the plaintext.

### 4. Encryption Pipeline

* **Spatial Decorrelation**: Logistic chaotic permutation of image pixels.
* **Encryption**: Standard **AES-256-GCM** with a 96-bit IV derived from HKDF.

## Validation Methodology

A maximalist validation suite (`analysis/validate_research.py`) characterizes system behavior without claiming cryptographic novelty.

### Entropy Source Behavior

* Mean motion-activity score: ~0.50 (active regime)
* Lag-1 autocorrelation: ~0.10
* Estimated min-entropy: ~2.18 bits/sample (upper bound)
* Gate pass rate (dynamic scenes): >98%
* Gate pass rate (static/dark scenes): <15%

### Cryptographic Sensitivity

* **NPCR**: ~99.6% (ideal >99.6%)
* **UACI**: ~33.5% (ideal ~33.4%)

### Ciphertext Statistics

* Adjacent pixel correlation: ≈ 0
* Histogram uniformity: chi-square p-value > 0.05

### Key Diversity & Stability

* Average Hamming distance between successive keys: ~50%
* Long-horizon analysis shows sustained activity without collapse into static behavior

## Repository Structure

```
.
├── analysis/
│   ├── analytics.py
│   ├── evaluate.py
│   └── validate_research.py
├── core/
│   ├── capture.py
│   ├── entropy.py
│   ├── conditioner.py
│   ├── kdf_aes.py
│   └── utils.py
├── data/
│   ├── lava_lamp_video.mp4
│   └── test_image.png
├── docs/
│   └── CVReportFinal.pdf
├── results/                 # Generated plots and CSVs
├── demo_layers.py           # Feature-level visualization
├── demo_live.py             # Real-time CV + entropy demo
├── main.py                  # End-to-end encryption entry point
├── requirements.txt
└── README.md
```

## Demos and Visualization

### Live CV & Entropy Dashboard

Visualizes optical flow, corner detection, and the evolving entropy signal in real time.

```bash
python demo_live.py
```

### Feature Layer Visualization

Displays individual entropy components over time.

```bash
python demo_layers.py
```

*All quantitative results reported in the paper are generated exclusively by the validation suite.*

## Requirements

Tested on **Python 3.8+**.

Key dependencies include `opencv-python`, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`, and `cryptography`.

```bash
pip install -r requirements.txt
```

## Usage

### Run Full Validation Suite

Generates all plots and statistical metrics in `results/`.

```bash
python analysis/validate_research.py
```

### Encrypt an Image

Uses recorded entropy to encrypt `data/test_image.png`.

```bash
python main.py
```

## Threat Model and Limitations

This project is a **research prototype**, not a certified hardware or software RNG.

Limitations include:

* No NIST SP 800-90B certification
* Empirical, application-specific entropy estimation
* Dependence on camera quality and environmental conditions
* No adversarial modeling of sensor manipulation

The system is designed to **fail safely**: when motion is absent, entropy collapses toward zero and generation halts.

## Citation

```bibtex
@misc{vision_entropy_pipeline,
  title  = {Vision-Driven Physical Entropy Extraction for Image Encryption},
  author = {Arnav Naik, Srujan R, Mayank Pavuskar},
  year   = {2025},
  note   = {Research prototype}
}
```

## License

MIT License. See `LICENSE` for details.
