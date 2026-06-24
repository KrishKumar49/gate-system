import cv2
import time
import os
import sys

from modules.exit.face_exit import verify_face
from modules.exit.vehicle_exit import verify_vehicle
from modules.exit.plate_exit import verify_plate

from modules.gate_logic.decision_engine import calculate_gate_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import get_active_vehicle_records, complete_visit, save_exit_record


def start_exit_monitoring():

    cap = cv2.VideoCapture("https://ik.imagekit.io/6f8hdxg1w/WhatsApp%20Video%202026-06-21%20at%2011.35.22%20PM.mp4?updatedAt=1782212509538")
    print("Opened: ", cap.isOpened())

    CACHE_REFRESH_INTERVAL = 5 
    SESSION_TIMEOUT = 10

    session_buffer = {}  

    last_cache_refresh = 0
    active_vehicles_records = []
    processed_visits = set()

    while True:
        if time.time() - last_cache_refresh > CACHE_REFRESH_INTERVAL:
            active_vehicles_records = get_active_vehicle_records()
            last_cache_refresh = time.time()
            
        ret, frame = cap.read()
        print("Frame Read: ", ret)
        if not ret:
            print("Video Finished")
            break
        
        face_result = verify_face(frame)
        vehicle_result = verify_vehicle(frame, active_vehicles_records)
        plate_result = verify_plate(frame, active_vehicles_records)
        
        print("FACE:", face_result)
        print("VEHICLE:", vehicle_result)
        print("PLATE:", plate_result)
        
        start = time.time()
        print("Frame processing", time.time() - start)
        if (
            face_result and
            face_result.get("verified") and
            face_result.get("visit_id")
        ):
            visit_id = face_result["visit_id"]
            
            if visit_id not in session_buffer:
                session_buffer[visit_id] = {
                    "face": None,
                    "vehicle": None,
                    "plate": None,
                    "last_seen": time.time()
                }
                
            session_buffer[visit_id]["face"] = face_result
            session_buffer[visit_id]["last_seen"] = time.time()
            
        if (
            vehicle_result and
            vehicle_result.get("verified") and
            vehicle_result.get("visit_id")
        ):
            visit_id = vehicle_result["visit_id"]
            if visit_id not in session_buffer:
                session_buffer[visit_id] = {
                    "face": None,
                    "vehicle": None,
                    "plate": None,
                    "last_seen": time.time()
                }
            session_buffer[visit_id]["vehicle"] = vehicle_result
            session_buffer[visit_id]["last_seen"] = time.time()

        if (
            plate_result and
            plate_result.get("verified") and
            plate_result.get("visit_id")
        ):
            visit_id = plate_result["visit_id"]
            if visit_id not in session_buffer:
                session_buffer[visit_id] = {
                    "face": None,
                    "vehicle": None,
                    "plate": None,
                    "last_seen": time.time()
                }
            session_buffer[visit_id]["plate"] = plate_result
            session_buffer[visit_id]["last_seen"] = time.time()
            
        current_time = time.time()
        stale_sessions = []
        
        for visit_id, data in session_buffer.items():
            if (
                current_time - 
                data["last_seen"]
            ) > SESSION_TIMEOUT:
                stale_sessions.append(visit_id)

        for visit_id in stale_sessions:
            del session_buffer[visit_id]
            
            
        for visit_id, data in list(session_buffer.items()):
            print(session_buffer)
            if (
                data["face"] is not None and
                data["vehicle"] is not None and
                data["plate"] is not None
            ):
                approved, score, matched_visit = calculate_gate_score(
                    data["face"],
                    data["vehicle"],
                    data["plate"]
                ) 
                
                if matched_visit in processed_visits:
                    del session_buffer[visit_id]
                    continue
                
                
                if approved:
                    print(f"EXIT APPROVED | "
                            f"Visit ID: {matched_visit} | "
                            f"Score: {score:.2f}"
                    )
                    
                    try:
                        processed_visits.add(matched_visit)
                        
                        print("CALLING save_exit_record")
                        
                        save_exit_record(
                            visit_id=matched_visit,
                            employee_id=data["face"].get("employee_id"),
                            vehicle_embedding=None,
                            vehicle_class=None,
                            plate_number=data["plate"].get("plate_number"),
                            face_verified=True,
                            vehicle_verified=True,
                            plate_verified=True,
                            gate_opened=True,
                            camera_id="exit_gate_cam"
                        )
                        
                        print("CALLING complete_visit")
                        complete_visit(matched_visit)
                    
                    except Exception as e:
                        print(f"Error saving exit record: {e}")
                else:
                    print(f"EXIT DENIED | "
                            f"Visit ID: {matched_visit} | "
                            f"Score: {score:.2f}"
                    )
                    
                del session_buffer[visit_id]
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()   
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    start_exit_monitoring()