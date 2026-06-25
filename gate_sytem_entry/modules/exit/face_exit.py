import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from database import get_active_employee_embeddings
from modules.enroll.enroll import get_face_app

MATCH_THRESHOLD = 0.40

def verify_face(frame):
    model = get_face_app()
    faces = model.get(frame)
    
    if not faces:
        return {
            "verified": False,
            "score": 0.0,
            "message": "No face detected"
        }

    
    
    live_embedding = faces[0].normed_embedding
    
    active_rows = get_active_employee_embeddings()
    
    best_score = 0.0
    best_employee = None
    best_visit = None
    
    
    for employee_id, visit_id, stored_embedding in active_rows:
        stored_emb = np.array(stored_embedding) if isinstance(stored_embedding, list) else np.array(eval(stored_embedding))
        similarity = np.dot(live_embedding, stored_emb)
        if similarity > best_score:
            best_score = similarity
            best_employee = employee_id
            best_visit = visit_id
            
    
    print(
        "FACE SCORE",
        best_employee,
        best_visit,
        best_score
    )
            
            
    if best_score > MATCH_THRESHOLD:
        return {
            "verified": True,
            "employee_id": best_employee,
            "visit_id": best_visit,
            "score": float(best_score),
            "message": f"Face verified with score {best_score:.2f}",
            "bbox": list(map(int, faces[0].bbox))
        }
        
    return {
        "verified": False,
        "score": float(best_score),
        "employee_id": best_employee,
        "visit_id": best_visit,
        "message": f"Face not verified. Best score: {best_score:.2f}"
    }