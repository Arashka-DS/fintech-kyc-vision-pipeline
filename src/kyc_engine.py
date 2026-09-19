import cv2
import numpy as np
import easyocr
import insightface
from ultralytics import YOLO
from insightface.app import FaceAnalysis

class KYCVisionEngine:
    def __init__(self):
        # 1. Initialize YOLOv8 (Pre-trained on document boundaries)
        # Note: In production, you would train a lightweight YOLOv8n specifically on Cart Melli/Bank Cards
        self.yolo_model = YOLO("yolov8n.pt") 
        
        # 2. Initialize EasyOCR (Persian + English for IBANs)
        self.reader = easyocr.Reader(['fa', 'en'], gpu=True)
        
        # 3. Initialize InsightFace (ArcFace for facial embeddings)
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Persian to English digit normalization map
        self.persian_digits = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')

    def check_image_quality(self, image_cv, threshold=100.0):
        """Variance of Laplacian to detect blurry or heavily compressed uploads."""
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score, blur_score > threshold

    def validate_national_code(self, code: str) -> bool:
        """Deterministic Modulo 11 Validation for Iranian National ID."""
        if not len(code) == 10 or not code.isdigit():
            return False
        
        digits = [int(x) for x in code]
        check_sum = sum(digits[i] * (10 - i) for i in range(9)) % 11
        
        if check_sum < 2:
            return digits[9] == check_sum
        else:
            return digits[9] == 11 - check_sum

    def extract_text_and_validate(self, cropped_img):
        """Runs OCR, normalizes Persian text, and hunts for National Codes via Regex."""
        results = self.reader.readtext(cropped_img, detail=0)
        raw_text = " ".join(results).translate(self.persian_digits)
        
        import re
        # Look for a 10-digit sequence
        potential_codes = re.findall(r'\b\d{10}\b', raw_text)
        
        for code in potential_codes:
            if self.validate_national_code(code):
                return code, True
        return None, False

    def face_match(self, id_image, selfie_image, threshold=0.6):
        """Extracts 512-D embeddings and computes Cosine Similarity."""
        faces_id = self.face_app.get(id_image)
        faces_selfie = self.face_app.get(selfie_image)

        if not faces_id or not faces_selfie:
            return 0.0, False # Could not detect faces in one or both images

        # Get embeddings for the most prominent face in each image
        emb_id = faces_id[0].normed_embedding
        emb_selfie = faces_selfie[0].normed_embedding

        # Cosine Similarity
        similarity = np.dot(emb_id, emb_selfie)
        return float(similarity), similarity > threshold

    def process_kyc_payload(self, id_img_bytes, selfie_img_bytes):
        # Decode bytes to OpenCV arrays
        id_img = cv2.imdecode(np.frombuffer(id_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        selfie_img = cv2.imdecode(np.frombuffer(selfie_img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 1. Quality Check
        blur_score, is_sharp = self.check_image_quality(id_img)
        if not is_sharp:
            return {"status": "REJECTED", "reason": f"ID image too blurry (Score: {blur_score:.2f})"}

        # 2. YOLO Document Cropping (Simulation)
        # In a real scenario, use YOLO bounding boxes to crop. 
        # Here we assume the full image if YOLO doesn't detect a specific sub-boundary.
        cropped_id = id_img 
        
        # 3. OCR & Deterministic Validation
        national_code, is_valid_id = self.extract_text_and_validate(cropped_id)
        if not is_valid_id:
            return {"status": "REJECTED", "reason": "Could not extract a valid 10-digit National Code."}

        # 4. Facial Similarity
        sim_score, is_match = self.face_match(id_img, selfie_img)

        return {
            "status": "APPROVED" if is_match else "MANUAL_REVIEW",
            "national_code": national_code,
            "face_similarity_score": round(sim_score, 4),
            "is_face_match": is_match,
            "image_clarity_score": round(blur_score, 2)
        }
