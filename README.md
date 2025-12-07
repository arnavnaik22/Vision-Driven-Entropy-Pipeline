# Random Motion-Based Image Encryption

## 👁️ Project Overview
This project implements a **True Random Number Generator (TRNG)** using Computer Vision to encrypt images. Unlike standard pseudo-random number generators (PRNGs) which are deterministic, this system extracts entropy from **real-world physical motion**—such as fluid dynamics, lava lamps, or hand movements—to generate unpredictable cryptographic keys.

The system captures video frames, analyzes chaotic motion features (optical flow, color drift), and synthesizes a high-entropy seed to drive **AES-256-GCM** encryption.

## 🎥 Inspiration
This project was inspired by the concept of physical entropy, famously used in **Cloudflare's Lava Lamp Wall** to secure the internet.

* **Concept Video:** [Tom Scott - The Lava Lamps That Help Keep The Internet Secure](https://www.youtube.com/watch?v=zlhsrRqttV4)
* **Full Project Report:** [CVReportFinal.pdf](CVReportFinal.pdf)

## 🛠️ Methodology
The system follows a 6-stage pipeline as detailed in the project report:

1.  **Motion Capture (`capture.py`):** Captures video frames using OpenCV (sampling every 3rd frame to reduce redundancy).
2.  **Entropy Extraction (`entropy.py`):** Quantifies chaos using:
    * **Optical Flow:** Dense motion vectors (Farnebäck algorithm).
    * **Color Drift:** HSV/Lab color space shifts between frames.
    * **Centroid Displacement:** Tracking moving blobs.
3.  **Conditioning (`conditioner.py`):** Aggregates entropy scores and hashes them using **SHA-256** to remove bias.
4.  **Key Generation (`kdf_aes.py`):** Uses **HKDF** (HMAC-based Key Derivation Function) to expand the seed into a secure AES key and IV.
5.  **Encryption:** Encrypts the target image using **AES-256 in GCM Mode**.
6.  **Evaluation (`evaluate.py`):** Validates security using NPCR (Number of Pixels Change Rate) and UACI metrics.

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `encrypt_demo.py` | **Main Entry Point.** Runs the full capture-to-encryption pipeline. |
| `capture.py` | Handles webcam video acquisition and frame preprocessing. |
| `entropy.py` | Algorithms for extracting entropy (Optical Flow, Shannon Entropy, LBP). |
| `conditioner.py` | Hashes and whitens the raw entropy stream. |
| `kdf_aes.py` | Handles AES-GCM encryption and Key Derivation. |
| `evaluate.py` | Calculates NPCR and UACI randomness metrics. |
| `analytics.py` | Generates plots for entropy variation across frames. |
| `utils.py` | Helper functions for image processing and file handling. |
| `test_image.png` | Sample input image for encryption. |

## 🧪 Test Data
The experimental results in the report were verified using footage from the following source. To reproduce the exact results, download this video and use it as the input:

* **Source Video:** [The Lava Lamps That Help Keep The Internet Secure (YouTube)](https://www.youtube.com/watch?v=zlhsrRqttV4)
* **Instructions:** Download the video, save it as `lava_lamp.mp4` in the `data/` folder, and run the script pointing to it.

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Encryption Demo:**
    * **Option A (Webcam):**
        ```bash
        python encrypt_demo.py
        ```
    * **Option B (Video File):**
        ```bash
        python encrypt_demo.py --video data/lava_lamp.mp4
        ```

3.  **View Results:**
    The script will generate an encrypted image and entropy graphs in the `results/` folder.

## 📊 Performance Results
Based on experimental trials:
* **NPCR (Number of Pixels Change Rate):** 99.60% (Ideal > 99%)
* **UACI (Unified Average Changing Intensity):** 49.92% (Ideal ≈ 33-50%)
* **Entropy Stability:** High-entropy peaks correspond to rapid color transitions and fluid flow.

## 🔗 References
* **NIST SP 800-90B:** Guidelines for Entropy Source Validation.
* **FIPS 197:** Advanced Encryption Standard (AES).
* **Cloudflare LavaRand:** Real-world implementation.

---
