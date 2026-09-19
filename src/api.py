import re
from fastapi import FastAPI, File, UploadFile
from src.kyc_engine import KYCVisionEngine

app = FastAPI(title="FinTech KYC Vision API", version="1.0.0")

# Initialize heavy ML models once on startup
engine = KYCVisionEngine()

@app.post("/verify-identity")
async def verify_identity(id_card: UploadFile = File(...), selfie: UploadFile = File(...)):
    id_bytes = await id_card.read()
    selfie_bytes = await selfie.read()
    
    # 1. Anti-Spoofing (FFT Moiré Detection)
    liveness_metrics = engine.detect_moire_fft(id_bytes)
    
    # Fail fast if spoofing is detected to save heavy GPU/OCR compute
    if liveness_metrics["is_screen_spoof"]:
        return {
            "status": "REJECTED_SPOOFING",
            "liveness": liveness_metrics,
            "message": "Digital screen replay detected. Please capture a live photo of the physical ID."
        }

    # 2. Biometric Facial Similarity (ArcFace)
    biometric_metrics = engine.compare_faces(id_bytes, selfie_bytes)
    
    if not biometric_metrics["match"]:
         return {
            "status": "REJECTED_BIOMETRIC_MISMATCH",
            "liveness": liveness_metrics,
            "biometrics": biometric_metrics,
            "message": "Selfie does not match the ID card photograph."
        }

    # 3. Document OCR & Modulo 11 Validation
    ocr_results = engine.reader.readtext(id_bytes)
    extracted_id = None
    
    for (bbox, text, prob) in ocr_results:
        # Strip non-numeric characters (handles Persian/Arabic/English digits due to OCR normalization)
        cleaned_text = re.sub(r'\D', '', text)
        
        # Check if it is a 10-digit string and passes the Modulo 11 checksum
        if len(cleaned_text) == 10 and engine.validate_iranian_national_id(cleaned_text):
            extracted_id = cleaned_text
            break

    if not extracted_id:
        return {
            "status": "REJECTED_DOCUMENT_INVALID",
            "liveness": liveness_metrics,
            "biometrics": biometric_metrics,
            "message": "Could not extract a valid Iranian National ID. Ensure the image is clear and glare-free."
        }

    # 4. Final Approval
    return {
        "status": "APPROVED",
        "national_id": extracted_id,
        "liveness": liveness_metrics,
        "biometrics": biometric_metrics,
        "message": "KYC onboarding successful."
    }

@app.get("/health")
def health_check():
    return {"status": "ACTIVE", "models_loaded": True}
