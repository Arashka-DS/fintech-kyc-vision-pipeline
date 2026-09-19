# Enterprise FinTech KYC & AML Vision Pipeline

A production-grade computer vision pipeline designed to automate Know Your Customer (KYC) onboarding for regional FinTech operations. This system reduces manual review hours while actively defending against identity spoofing and digital forgery.

## Architecture & Security Layers
1. **Forgery Detection (EXIF):** Inspects image metadata to reject software-manipulated uploads (e.g., Adobe Photoshop).
2. **Liveness & Presentation Attack Detection:** Utilizes Fast Fourier Transform (FFT) to detect high-frequency Moiré patterns, instantly flagging attackers attempting to bypass the system by pointing their camera at a digital screen.
3. **Variance of Laplacian:** Calculates pixel variance to reject blurry, unreadable documents before wasting GPU compute.
4. **Deterministic Modulo 11:** OCR (EasyOCR) extracts text, normalizes Persian digits, and runs a mathematical checksum to guarantee the Iranian National Code is valid.
5. **ArcFace Similarity:** `InsightFace` computes the 512-D Cosine Similarity between the ID document and the live selfie.
6. **Regulatory Audit:** `FastAPI` background tasks stream the decision matrix into `PostgreSQL` for AML compliance and `Metabase` visualization.

## Quick Start
1. `mamba env create -f environment.yml && mamba activate kyc_vision_env`
2. `docker-compose up -d --build`
3. Send a POST request with `id_card` and `selfie` images via multipart form data to `http://localhost:8000/verify-identity`.
4. Open Metabase at `http://localhost:3000` to monitor the SOC manual review queue and spoofing alerts in real-time.
