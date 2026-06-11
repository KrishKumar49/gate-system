import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import get_active_vehicle_records
from modules.v_e_local.vehicle_entry import detector, get_vehicle_embedding

MATCH_THRESHOLD = 0.70

def verify_vehicle(frame):
    results = detector.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]
    
    if (
        results.boxes is None or
        results.boxes.id is None
    ):
        return None
    
    best_score = 0.0
    best_match = None
    
    box_ids = results.boxes.id.cpu().numpy()
    cls_indices = results.boxes.cls.cpu().numpy()
    
    active_vehicles = get_active_vehicle_records()
    
    if not active_vehicles:
        return {
            "verified": False,
            "employee_id": None,
            "visit_id": None,
            "score": 0.0,
            "plate_number": None
        }
    
    for idx, track_id in enumerate(box_ids):
        class_name = results.names[cls_indices[idx]]
        if class_name != "motorcycle" :
            continue
        
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx].cpu().numpy())
        
        crop = frame[
            max(0, y1): y2,
            max(0, x1): x2
        ]
        
        live_embedding = get_vehicle_embedding(crop)
        
        if live_embedding is None:
            continue
        
        for employee_id, visit_id, stored_embedding, plate_number in active_vehicles:
            
            if stored_embedding is None:
                continue
            
            stored_emb = np.array(stored_embedding, dtype=np.float32)
            similarity = np.dot(live_embedding, stored_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "employee_id": employee_id,
                    "visit_id": visit_id,
                    "plate_number": plate_number,
                    "score": float(similarity),
                    "bbox": [x1, y1, x2, y2]
                }
                
    if best_score > MATCH_THRESHOLD:
        return {
            "verified": True,
            **best_match,
        }
        
    return {
    "verified": False,
    "employee_id": None,
    "visit_id": None,
    "score": float(best_score),
    "plate_number": None
}
    
             