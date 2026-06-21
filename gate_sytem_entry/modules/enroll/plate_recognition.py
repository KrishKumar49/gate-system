import argparse
import os
import sys
from collections import Counter, deque
import cv2
import numpy as np
from fast_alpr import ALPR
try:
    from PIL import Image
except Exception:
    Image = None

# Ensure database access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from database import log_plate_recognition

FRAME_SKIP_DEFAULT = 5
alpr = ALPR()
CONFIDENCE_THRESHOLD = 0.45
VOTING_WINDOW_SIZE = 10





def run_plate_logic(frame_queue, entry_id_queue, ready_event=None):
    print("plate recognition worker started")
    # signal readiness to the parent process
    if ready_event is not None:
        try:
            print("Plate worker ready")
            ready_event.set()
        except Exception:
            pass
    plate_votes = Counter()
    plate_window = deque()
    last_logged_plate = None
    current_entry_id = None  # This should be set to the current visit/entry context ID when available
    
    def _confidence_value(conf):
        if isinstance(conf, (list, tuple, np.ndarray)):
            return float(np.mean(conf))
        return float(conf)
    
    def _normalize_plate(text):
        if not text:
            return ''
        return ''.join(ch for ch in text.strip().upper() if ch.isalnum())
    
    while True:  
        try:
            if not entry_id_queue.empty():
                current_entry_id = entry_id_queue.get_nowait()
                plate_votes.clear()
                plate_window.clear()
                last_logged_plate = None
                print(f"Switched to new entry context: {current_entry_id}")
        except Exception as e:
            pass
                      
        frame = frame_queue.get()
        print("Plate worker received frame")
        if frame is None: break
        
        detections = alpr.predict(frame)
        print("Plate detections:", len(detections))
        for det in detections:
            ocr = getattr(det, 'ocr', None)
            txt = getattr(ocr, 'text', '') if ocr is not None else ''
            conf = _confidence_value(getattr(ocr, 'confidence', 0)) if ocr is not None else 0
            
            if conf >= CONFIDENCE_THRESHOLD and txt:
                plate_number = _normalize_plate(txt)
                
                plate_window.append(plate_number)
                plate_votes[plate_number] += 1
                
                if len(plate_window) > VOTING_WINDOW_SIZE:
                    old = plate_window.popleft()
                    plate_votes[old] -= 1
                    if plate_votes[old] <= 0:
                        del plate_votes[old]
                
                if current_entry_id and plate_votes[plate_number] >= 5 and plate_number != last_logged_plate:
                    print(f"Detected plate: {plate_number} with confidence {conf}")
                    log_plate_recognition(current_entry_id, plate_number)  # Log to database or file as needed 
                    last_logged_plate = plate_number
                    
                    
                    
                