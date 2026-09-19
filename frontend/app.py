import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="FinTech KYC Portal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card { background-color: #111827; padding: 20px; border-radius: 10px; border: 1px solid #1F2937; }
    .status-approved { color: #00E5FF; font-weight: bold; font-size: 24px;}
    .status-rejected { color: #FF1744; font-weight: bold; font-size: 24px;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Automated KYC & Identity Verification")
st.markdown("Upload a National ID (Cart Melli) and a live selfie to evaluate spoofing, OCR extraction, and biometric similarity.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. National ID Upload")
    id_file = st.file_uploader("Upload ID Document", type=['jpg', 'jpeg', 'png'], key="id")

with col2:
    st.subheader("2. Live Selfie Upload")
    selfie_file = st.file_uploader("Upload User Selfie", type=['jpg', 'jpeg', 'png'], key="selfie")

if id_file and selfie_file:
    # Display the uploaded images
    img_col1, img_col2 = st.columns(2)
    img_col1.image(Image.open(id_file), caption="Document Scan", use_column_width=True)
    img_col2.image(Image.open(selfie_file), caption="Biometric Selfie", use_column_width=True)

    if st.button("Run Compliance Engine", type="primary", use_container_width=True):
        with st.spinner("Analyzing pixels, extracting embeddings, and checking EXIF data..."):
            try:
                # Reset file pointers
                id_file.seek(0)
                selfie_file.seek(0)
                
                files = {
                    "id_card": (id_file.name, id_file.getvalue(), "image/jpeg"),
                    "selfie": (selfie_file.name, selfie_file.getvalue(), "image/jpeg")
                }
                
                res = requests.post("http://kyc_api:8000/verify-identity", files=files)
                res.raise_for_status()
                data = res.json()
                
                st.divider()
                st.subheader("Engine Decision Matrix")
                
                # Top Level Decision
                if data['status'] == "APPROVED":
                    st.markdown(f"<div class='status-approved'>✅ {data['status']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='status-rejected'>🚨 {data['status']}: {data.get('reason', 'Verification Failed')}</div>", unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                
                m1.metric("Extracted National Code", data.get('national_code', 'FAILED'))
                m2.metric("Face Similarity", f"{data.get('face_similarity_score', 0.0) * 100:.1f}%")
                m3.metric("Image Clarity", data.get('image_clarity_score', 0.0))
                
                spoof_color = "normal" if not data.get('is_spoof') else "inverse"
                m4.metric("Spoof Detected", str(data.get('is_spoof', False)), delta_color=spoof_color)

            except Exception as e:
                st.error(f"Engine connection failed: {e}")
