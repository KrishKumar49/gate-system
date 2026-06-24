import cv2
import numpy as np
import os
import sys
import time

import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import get_active_vehicle_records
from modules.v_e_local.vehicle_entry import detector, get_vehicle_embedding

MATCH_THRESHOLD = 0.70
TRACK_TIMEOUT = 15 
MATCH_THRESHOLD = 0.70

TRACK_CACHE = {}  # {track_id: {"last_seen": timestamp, "result": {...}}}

def verify_vehicle(frame, active_vehicles):
    current_time = time.time()
    
    expired_tracks = []
    
    for tid, data in TRACK_CACHE.items():
        if current_time - data['last_seen'] > TRACK_TIMEOUT:
            expired_tracks.append(tid)
            
    for tid in expired_tracks:
        del TRACK_CACHE[tid]
    
    
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
    
    # active_vehicles = get_active_vehicle_records()
    
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
        
        # live_embedding = get_vehicle_embedding(crop)
        
        # if live_embedding is None:
        #     continue
        
        track_id = int(track_id)
        if (
            track_id in TRACK_CACHE
            and current_time - TRACK_CACHE[track_id]['last_seen'] < TRACK_TIMEOUT
        ):
            TRACK_CACHE[track_id]['last_seen'] = current_time
            
            print(
                "CACHE HIT",
                track_id,
                TRACK_CACHE[track_id]['result']['visit_id']
            )
            
            return {
                "verified": True,
                **TRACK_CACHE[track_id]['result']
            }
            
        live_embedding = get_vehicle_embedding(crop)
        if live_embedding is None:
            continue
        
        
        for employee_id, visit_id, stored_embedding, plate_number in active_vehicles:
            
            if stored_embedding is None:
                continue
            
            # print(type(stored_embedding))
            # print(stored_embedding)
            
            stored_emb = np.array(
                ast.literal_eval(stored_embedding),
                dtype=np.float32
            )
            
            
            similarity = np.dot(live_embedding, stored_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "track_id": int(track_id),
                    "employee_id": employee_id,
                    "visit_id": visit_id,
                    "plate_number": plate_number,
                    "score": float(similarity),
                    "bbox": [x1, y1, x2, y2]
                }
                
    if best_score > MATCH_THRESHOLD:
        result = {
            "track_id": best_match["track_id"],
            "employee_id": best_match["employee_id"],
            "visit_id": best_match["visit_id"],
            "plate_number": best_match["plate_number"],
            "score": float(best_score),
            "bbox": best_match["bbox"]
        }
        
        TRACK_CACHE[best_match["track_id"]] = {
            "last_seen": current_time,
            "result": result
        }
        
        return {
            "verified": True,
            **result
        }
        
    return {
    "verified": False,
    "employee_id": None,
    "visit_id": None,
    "score": float(best_score),
    "plate_number": None
}
    
             