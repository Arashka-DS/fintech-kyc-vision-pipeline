from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from src.kyc_engine import KYCVisionEngine
import psycopg2
import os

app = FastAPI(title="FinTech Automated KYC Pipeline")
kyc_engine = KYCVisionEngine()

def log_audit_to_db(result: dict):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database="kyc_ledger", user="kyc_admin", password="kyc_password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO kyc_audit_log 
            (national_code, status, rejection_reason, face_similarity_score, image_clarity_score, is_spoof_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            result.get('national_code', 'N/A'), result['status'], result.get('reason', ''), 
            result.get('face_similarity_score', 0.0), result.get('image_clarity_score', 0.0), result.get('is_spoof', False)
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Logging Error: {e}")

@app.post("/verify-identity")
async def verify_identity(
    bg_tasks: BackgroundTasks,
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    try:
        id_bytes = await id_card.read()
        selfie_bytes = await selfie.read()
        
        result = kyc_engine.process_kyc_payload(id_bytes, selfie_bytes)
        
        # Non-blocking audit log
        bg_tasks.add_task(log_audit_to_db, result)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
