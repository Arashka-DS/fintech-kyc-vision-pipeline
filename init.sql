CREATE TABLE IF NOT EXISTS kyc_audit_log (
    audit_id SERIAL PRIMARY KEY,
    national_id VARCHAR(20),
    status VARCHAR(50),
    fft_liveness_score NUMERIC(8, 2),
    face_similarity_score NUMERIC(5, 4),
    message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
