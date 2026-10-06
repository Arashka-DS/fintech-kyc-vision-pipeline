CREATE TABLE IF NOT EXISTS kyc_audit_log (
    audit_id SERIAL PRIMARY KEY,
    national_id VARCHAR(15),
    status VARCHAR(20),
    fft_liveness_score NUMERIC,
    face_similarity_score NUMERIC,
    rejection_reason VARCHAR(100),
    is_spoof_detected BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
