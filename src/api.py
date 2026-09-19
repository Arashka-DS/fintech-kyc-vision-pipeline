import re
import os
import psycopg2
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from src.kyc_engine import KYCVisionEngine

app = FastAPI(title="FinTech KYC Vision API", version="1.1.0")
engine = KYCVisionEngine()

def log_audit_to_db(national_id: str, status: str, fft_score: float, sim_score: float, message: str):
    """Background task to asynchronously stream KYC verdicts to PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "kyc_warehouse"),
            user=os.getenv("DB_USER", "kyc_admin"),
            password=os.getenv("DB_PASSWORD", "kyc_password"),
            port=5432
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kyc_audit_log (
                audit_id SERIAL PRIMARY KEY,
                national_id VARCHAR(20),
                status VARCHAR(50),
                fft_liveness_score NUMERIC(8, 2),
                face_similarity_score NUMERIC(5, 4),
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            INSERT INTO kyc_audit_log (national_id, status, fft_liveness_score, face_similarity_score, message)
            VALUES (%s, %s, %s, %s, %s)
        """, (national_id, status, fft_score, sim_score, message))
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Database Logging Error: {e}")

@app.post("/verify-identity")
async def verify_identity(background_tasks: BackgroundTasks, id_card: UploadFile = File(...), selfie: UploadFile = File(...)):
    id_bytes = await id_card.read()
    selfie_bytes = await selfie.read()
    
    # 1. Anti-Spoofing
    liveness_metrics = engine.detect_moire_fft(id_bytes)
    fft_score = liveness_metrics["fft_high_freq_energy"]
    
    if liveness_metrics["is_screen_spoof"]:
        status, msg = "REJECTED_SPOOFING", "Digital screen replay detected."
        background_tasks.add_task(log_audit_to_db, "UNKNOWN", status, fft_score, 0.0, msg)
        return {"status": status, "liveness": liveness_metrics, "message": msg}

    # 2. Biometric Similarity
    biometric_metrics = engine.compare_faces(id_bytes, selfie_bytes)
    sim_score = biometric_metrics["similarity_score"]
    
    if not biometric_metrics["match"]:
        status, msg = "REJECTED_BIOMETRIC_MISMATCH", "Selfie does not match ID."
        background_tasks.add_task(log_audit_to_db, "UNKNOWN", status, fft_score, sim_score, msg)
        return {"status": status, "liveness": liveness_metrics, "biometrics": biometric_metrics, "message": msg}

    # 3. Document OCR & Modulo 11 
    ocr_results = engine.reader.readtext(id_bytes)
    extracted_id = None
    
    for (bbox, text, prob) in ocr_results:
        cleaned_text = re.sub(r'\D', '', text)
        if len(cleaned_text) == 10 and engine.validate_iranian_national_id(cleaned_text):
            extracted_id = cleaned_text
            break

    if not extracted_id:
        status, msg = "REJECTED_DOCUMENT_INVALID", "Could not extract a valid Iranian National ID."
        background_tasks.add_task(log_audit_to_db, "UNKNOWN", status, fft_score, sim_score, msg)
        return {"status": status, "liveness": liveness_metrics, "biometrics": biometric_metrics, "message": msg}

    # 4. Final Approval
    status, msg = "APPROVED", "KYC onboarding successful."
    background_tasks.add_task(log_audit_to_db, extracted_id, status, fft_score, sim_score, msg)
    
    return {
        "status": status,
        "national_id": extracted_id,
        "liveness": liveness_metrics,
        "biometrics": biometric_metrics,
        "message": msg
    }

@app.get("/health")
def health_check():
    return {"status": "ACTIVE", "models_loaded": True}
