-- KYC Funnel (Funnel Visual)

SELECT
  status,
  COUNT(*)
FROM kyc_audit_log
GROUP BY 1;
-- (Drop-off from Total Submissions Approved vs. Manual Review).
  
-- Spoofing & Fraud Attempts (Red Bar Chart)

SELECT 
  date_trunc('hour', timestamp),
  COUNT(*)
FROM kyc_audit_log
WHERE is_spoof_detected = TRUE
GROUP BY 1;


-- Live Manual Review Queue (Table)
SELECT
  audit_id,
  national_code,
  rejection_reason,
  timestamp
FROM kyc_audit_log
WHERE status = 'MANUAL_REVIEW'
ORDER BY timestamp DESC;
