from fastapi import FastAPI, UploadFile, File, HTTPException
from src.kyc_engine import KYCVisionEngine

app = FastAPI(title="FinTech Automated KYC Pipeline", version="2.0.0")

# Initialize Engine globally to keep models loaded in GPU memory
print("Booting Vision Models into VRAM...")
kyc_engine = KYCVisionEngine()

@app.post("/verify-identity")
async def verify_identity(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...)
):
    # Validate MIME types
    valid_types = ["image/jpeg", "image/png"]
    if id_card.content_type not in valid_types or selfie.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="Only JPG and PNG formats are supported.")

    try:
        # Read files into memory bytes
        id_bytes = await id_card.read()
        selfie_bytes = await selfie.read()
        
        # Execute pipeline
        result = kyc_engine.process_kyc_payload(id_bytes, selfie_bytes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KYC Pipeline Failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "CUDA Online", "models_loaded": True}
