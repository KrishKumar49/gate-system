from collections import Counter, deque
from difflib import SequenceMatcher
import numpy as np
import time

from modules.enroll.plate_recognition import alpr

CONFIDENCE_THRESHOLD = 0.75
VOTING_WINDOW_SIZE = 10

plate_votes = Counter()
plate_window = deque()

last_seen_plate_time = time.time()

def verify_plate(frame, active_vehicle_records):
    global last_seen_plate_time
    detections = alpr.predict(frame)
    
    if len(detections) == 0:
        if time.time() - last_seen_plate_time > 5:
            plate_votes.clear()
            plate_window.clear()
        return {
            "verified": False,
            "score": 0.0,
            "message": "No plates detected"
        }
    
    
    def normalize_plate(text):
        if not text:
            return ''
        return ''.join(
            ch
            for ch in text.strip().upper()
            if ch.isalnum()
        )
        
    for det in detections:
        ocr = getattr(det, 'ocr', None)
        if ocr is None:
            continue
        
        text = getattr(ocr, 'text', "")
        confidence = getattr(ocr, 'confidence', 0.0)
        
        if isinstance(confidence, (list, tuple, np.ndarray)):
            confidence = float(np.mean(confidence))
            
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        plate = normalize_plate(text)
        
        if not plate:
            continue
        
        last_seen_plate_time = time.time()
        
        plate_window.append(plate)
        plate_votes[plate] += 1
        
        if len(plate_window) > VOTING_WINDOW_SIZE:
            old_plate = plate_window.popleft()
            plate_votes[old_plate] -= 1
            if plate_votes[old_plate] <= 0:
                del plate_votes[old_plate]
        
    if not plate_votes:
        return {
            "verified": False,
            "score": 0.0,
            "message": "No plates detected"
        }
        
    best_plate = max(
        plate_votes,
        key=plate_votes.get
    )
    
    if plate_votes[best_plate] < 3:
        return {
            "verified": False,
            "score": 0.0,
            "plate_number": best_plate,
            "message": f"Plate {best_plate} detected but not stable enough yet"
        }
    
    best_score = 0.0
    best_match = None
    
    for employee_id, visit_id, _, stored_plate in active_vehicle_records:
        if not stored_plate:
            continue
        
        score = SequenceMatcher(
            None,
            normalize_plate(stored_plate),
            best_plate
        ).ratio()
        
        if score > best_score:
            best_score = score
            best_match = {
                "employee_id": employee_id,
                "visit_id": visit_id,
                "plate_number": stored_plate
            }

        if score >= 0.9 and plate_votes[best_plate] >= 3:
            return {
                "verified": True,
                "employee_id": employee_id,
                "visit_id": visit_id,
                "score": score,
                "plate_number": best_plate,
                "message": f"Plate {best_plate} verified with {plate_votes[best_plate]} votes"
            }

    return {
        "verified": False,
        "employee_id": best_match["employee_id"] if best_match else None,
        "visit_id": best_match["visit_id"] if best_match else None,
        "score": best_score,
        "plate_number": best_plate,
        "message": f"Plate {best_plate} not found in active records"
    }