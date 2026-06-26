# import argparse
# import os
# import sys
# from collections import Counter, deque
# from queue import Empty
# import cv2
# import numpy as np
# from fast_alpr import ALPR
# try:
#     from PIL import Image
# except Exception:
#     Image = None

# # Ensure database access
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from database import log_plate_recognition, update_latest_vehicle_plate

# FRAME_SKIP_DEFAULT = 5
# alpr = ALPR()
# CONFIDENCE_THRESHOLD = 0.45
# VOTING_WINDOW_SIZE = 10

# from queue import Empty


# def run_plate_logic(frame_queue, entry_id_queue, ready_event=None):
#     print("plate recognition worker started")
#     # signal readiness to the parent process
    
    
    
#     if ready_event is not None:
#         try:
#             print("Plate worker ready")
#             ready_event.set()
#         except Exception:
#             pass
#     plate_votes = Counter()
#     plate_window = deque()
#     best_plate = None
#     best_votes = 0
#     last_logged_plate = None
#     current_entry_id = None  # This should be set to the current visit/entry context ID when available
    
#     def _confidence_value(conf):
#         if isinstance(conf, (list, tuple, np.ndarray)):
#             return float(np.mean(conf))
#         return float(conf)
    
#     def _normalize_plate(text):
#         if not text:
#             return ''
#         return ''.join(ch for ch in text.strip().upper() if ch.isalnum())
    
    
#     while True:
        
#         frame = frame_queue.get()
#         print("Plate worker received frame")
#         if frame is None: break
        
#         detections = alpr.predict(frame)
#         print("Plate detections:", len(detections))
#         for det in detections:
#             ocr = getattr(det, 'ocr', None)
#             txt = getattr(ocr, 'text', '') if ocr is not None else ''
#             conf = _confidence_value(getattr(ocr, 'confidence', 0)) if ocr is not None else 0
            
#             print(
#                 "OCR: ", txt,
#                 "Confidence: ", conf,
#             )
            
            
            
#             if conf >= CONFIDENCE_THRESHOLD and txt:
#                 plate_number = _normalize_plate(txt)
                
#                 plate_window.append(plate_number)
#                 plate_votes[plate_number] += 1
                
#                 if plate_votes[plate_number] > best_votes:
#                     best_votes = plate_votes[plate_number]
#                     best_plate = plate_number
                
#                 print(
#                     "PLATE:",
#                     plate_number,
#                     "VOTES:",
#                     plate_votes[plate_number]
#                 )
                
#                 if len(plate_window) > VOTING_WINDOW_SIZE:
#                     old = plate_window.popleft()
#                     plate_votes[old] -= 1
#                     if plate_votes[old] <= 0:
#                         del plate_votes[old]
                        
#                 print(
#                     "BEST PLATE:",
#                     best_plate,
#                     "BEST VOTES:",
#                     best_votes
#                 )
                
#                 if best_plate and best_votes >= 2 and best_plate != last_logged_plate:

#                     print("SAVING PLATE")

#                     updated_id = update_latest_vehicle_plate(best_plate)
                    

#                     print(
#                         "UPDATED ENTRY:",
#                         updated_id,
#                         "PLATE:",
#                         best_plate
#                     )

#                     last_logged_plate = best_plate

#                     plate_votes.clear()
#                     plate_window.clear()
#                     best_plate = None
#                     best_votes = 0

#                     break
                
     
     
     
     
     
     
     
     
     
     
                
                
                
                
import argparse
import os
import sys
from collections import Counter, deque
from queue import Empty
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

from queue import Empty


def run_plate_logic(frame_queue, entry_id_queue, vehicle_done_queue, ready_event=None):
    print("plate recognition worker started")
    # signal readiness to the parent process
    plate_votes = Counter()
    plate_window = deque()
    best_plate = None
    best_votes = 0
    last_logged_plate = None
    current_entry_id = None  # This should be set to the current visit/entry context ID when available
    
    print("ALPR initialized:")
    
    if ready_event is not None:
        try:
            print("Plate worker ready")
            ready_event.set()
        except Exception:
            pass
    
    def _confidence_value(conf):
        if isinstance(conf, (list, tuple, np.ndarray)):
            return float(np.mean(conf))
        return float(conf)
    
    def _normalize_plate(text):
        if not text:
            return ''
        return ''.join(ch for ch in text.strip().upper() if ch.isalnum())
    
    
    while True:
        
        if current_entry_id is None:
            try:
                print("WAITING FOR ENTRY ID...")

                current_entry_id = entry_id_queue.get(timeout=10)

                print(
                    "ENTRY ID RECEIVED:",
                    current_entry_id
                )

            except Exception:
                print("NO ENTRY ID")
                continue
        
        frame = frame_queue.get()
        print("Plate worker received frame")
        if frame is None: break
        
        detections = alpr.predict(frame)
        print("Plate detections:", len(detections))
        for det in detections:
            ocr = getattr(det, 'ocr', None)
            txt = getattr(ocr, 'text', '') if ocr is not None else ''
            conf = _confidence_value(getattr(ocr, 'confidence', 0)) if ocr is not None else 0
            
            print(
                "OCR: ", txt,
                "Confidence: ", conf,
            )
            
            
            
            if conf >= CONFIDENCE_THRESHOLD and txt:
                plate_number = _normalize_plate(txt)
                
                plate_window.append(plate_number)
                plate_votes[plate_number] += 1
                
                if plate_votes[plate_number] > best_votes:
                    best_votes = plate_votes[plate_number]
                    best_plate = plate_number
                
                print(
                    "PLATE:",
                    plate_number,
                    "VOTES:",
                    plate_votes[plate_number]
                )
                
                if len(plate_window) > VOTING_WINDOW_SIZE:
                    old = plate_window.popleft()
                    plate_votes[old] -= 1
                    if plate_votes[old] <= 0:
                        del plate_votes[old]
                        
                print(
                    "BEST PLATE:",
                    best_plate,
                    "BEST VOTES:",
                    best_votes
                )
                
                if best_plate and best_votes >= 2 and best_plate != last_logged_plate:

                    print("SAVING PLATE")

                    updated_id = log_plate_recognition(
                        current_entry_id,
                        best_plate
                    )
                    
                    vehicle_done_queue.put(current_entry_id)
                    
                    

                    print(
                        "UPDATED ENTRY:",
                        updated_id,
                        "PLATE:",
                        best_plate
                    )
                    
                    current_entry_id = None

                    last_logged_plate = best_plate

                    plate_votes.clear()
                    plate_window.clear()
                    best_plate = None
                    best_votes = 0

                    break