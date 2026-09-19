CREATE TABLE IF NOT EXISTS kyc_audit_log (
    audit_id SERIAL PRIMARY KEY,
    national_code VARCHAR(15),
    status VARCHAR(20),
    rejection_reason VARCHAR(100),
    face_similarity_score NUMERIC,
    image_clarity_score NUMERIC,
    is_spoof_detected BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
