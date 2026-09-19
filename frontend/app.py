import streamlit as st
import requests

st.set_page_config(page_title="KYC & AML Compliance Portal", layout="wide")

API_URL = "http://kyc_api:8000"

st.title("🛡️ KYC Biometric Onboarding Portal")
st.markdown("Upload a physical National ID card and a live selfie. The system checks for digital screen spoofing (FFT), extracts & validates the National Code (Modulo 11), and verifies biometric similarity.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload National ID Card")
    id_upload = st.file_uploader("Upload ID (Front)", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("2. Upload Live Selfie")
    selfie_upload = st.file_uploader("Upload Selfie", type=["jpg", "jpeg", "png"])

if id_upload and selfie_upload:
    st.divider()
    
    if st.button("Run KYC Verification Pipeline", type="primary"):
        with st.spinner("Analyzing FFT frequencies, running OCR, and extracting facial embeddings..."):
            try:
                # Prepare multipart form data
                files = {
                    "id_card": (id_upload.name, id_upload.getvalue(), id_upload.type),
                    "selfie": (selfie_upload.name, selfie_upload.getvalue(), selfie_upload.type)
                }
                
                response = requests.post(f"{API_URL}/verify-identity", files=files)
                data = response.json()
                
                # Render Results
                status = data.get("status")
                
                if status == "APPROVED":
                    st.success(f"✅ **APPROVED:** Identity Verified. National ID: `{data.get('national_id')}`")
                else:
                    st.error(f"🚨 **REJECTED:** {data.get('message')}")
                
                # Render Telemetry
                st.subheader("Pipeline Telemetry")
                m1, m2, m3 = st.columns(3)
                
                liveness = data.get("liveness", {})
                biometrics = data.get("biometrics", {})
                
                # Liveness FFT Metric
                fft_score = liveness.get('fft_high_freq_energy', 0)
                fft_color = "normal" if fft_score < 145.0 else "inverse"
                m1.metric("Moiré FFT Energy", f"{fft_score}", delta="Threshold: 145", delta_color=fft_color)
                
                # Biometric Metric
                sim_score = biometrics.get('similarity_score', 0)
                sim_color = "normal" if sim_score >= 0.60 else "inverse"
                m2.metric("ArcFace Similarity", f"{sim_score}", delta="Threshold: 0.60", delta_color=sim_color)
                
                # Status Enum
                m3.metric("System Verdict", status)

            except Exception as e:
                st.error(f"Connection Error: Is the FastAPI backend running? Details: {e}")
