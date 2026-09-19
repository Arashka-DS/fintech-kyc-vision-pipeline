# FinTech KYC & Biometric Vision Pipeline

A production-grade multimodal computer vision pipeline designed to automate Know Your Customer (KYC) onboarding for regional FinTech operations. This system reduces manual review hours while actively defending against identity spoofing and digital forgery using advanced image processing heuristics.

## 🏛️ Architecture & Security Layers
1. **Liveness & Presentation Attack Detection:** Utilizes Fast Fourier Transform (FFT) to detect high-frequency Moiré patterns, instantly flagging attackers attempting to bypass the system by pointing their camera at a digital screen.
2. **Deterministic Modulo 11:** OCR (`EasyOCR`) extracts text, normalizes Persian/Arabic digits, and runs a mathematical checksum to guarantee the Iranian National Code is structurally valid.
3. **Biometric Similarity:** `InsightFace` computes the 512-D Cosine Similarity between the face detected on the ID document and the live selfie.
4. **Regulatory Audit:** `FastAPI` background tasks stream the decision matrix into `PostgreSQL` for AML compliance tracking.

## ⚙️ Stack
* **Environment:** `mamba` / `micromamba` (For deterministic resolution of CUDA/GPU binaries)
* **Computer Vision:** `OpenCV`, `InsightFace`, `EasyOCR`, `YOLOv8`
* **Infrastructure:** `FastAPI`, `Streamlit`, `PostgreSQL`, `Metabase`

## 🚀 Quick Start
1. **Boot the complete infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
   (Note: The build process uses `micromamba` to cleanly install heavy PyTorch and OpenCV dependencies).
2. **Access the Streamlit Inspection Portal:**
    Navigate to `http://localhost:8501` to upload an ID card and a selfie. The UI will render the FFT spoofing metrics and facial similarity scores in real-time.
3. **Access Operational BI:**
    Open Metabase at `http://localhost:3000` to monitor the SOC manual review queue and spoofing alerts.
