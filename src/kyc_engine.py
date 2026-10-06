import torch
import cv2
import numpy as np
import easyocr
from insightface.app import FaceAnalysis

class KYCVisionEngine:
    def __init__(self):
        # Dynamically detect hardware for container portability
        use_gpu = torch.cuda.is_available()
        
        # Initialize OCR for Persian/Arabic and English digits
        self.reader = easyocr.Reader(['fa', 'en'], gpu=use_gpu) 
        
        # Initialize ArcFace for biometric similarity
        self.face_app = FaceAnalysis(name='buffalo_l')
        ctx_id = 0 if use_gpu else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect_moire_fft(self, image_bytes: bytes) -> dict:
        """Uses Fast Fourier Transform (FFT) to detect digital screens (Moiré patterns)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = img.shape
        cy, cx = h // 2, w // 2
        
        # Mask out the low frequencies (the center)
        high_freq_mask = np.ones((h, w), np.uint8)
        cv2.circle(high_freq_mask, (cx, cy), 30, 0, -1)
        
        # Calculate energy metrics
        total_energy = np.mean(magnitude_spectrum)
        high_freq_energy = np.mean(magnitude_spectrum * high_freq_mask)
        
        # Create a lighting-invariant ratio
        ratio = float(high_freq_energy / (total_energy + 1e-7))
        
        # Standard ratio threshold for screen detection
        is_spoof = bool(ratio > 0.85) 
        
        return {
            "is_screen_spoof": is_spoof,
            "fft_high_freq_energy": round(ratio, 2)
        }
    def compare_faces(self, id_bytes: bytes, selfie_bytes: bytes) -> dict:
        """Calculates 512-D Cosine Similarity between the ID face and live selfie."""
        id_img = cv2.imdecode(np.frombuffer(id_bytes, np.uint8), cv2.IMREAD_COLOR)
        selfie_img = cv2.imdecode(np.frombuffer(selfie_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        faces_id = self.face_app.get(id_img)
        faces_selfie = self.face_app.get(selfie_img)
        
        if not faces_id or not faces_selfie:
            return {"match": False, "similarity_score": 0.0, "error": "Face not detected."}
            
        emb_id = faces_id[0].embedding
        emb_selfie = faces_selfie[0].embedding
        
        similarity = np.dot(emb_id, emb_selfie) / (np.linalg.norm(emb_id) * np.linalg.norm(emb_selfie))
        score = float(similarity)
        
        return {
            "match": score > 0.45, 
            "similarity_score": round(score, 4)
        }
