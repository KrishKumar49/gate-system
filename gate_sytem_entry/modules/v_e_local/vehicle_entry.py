import os
import sys
import logging
import importlib
import time

import cv2
import torch
import numpy as np

# Forces Python to discover the cloned fast-reid folder layout automatically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTREID_PATH = os.path.join(CURRENT_DIR, "fast-reid")
if FASTREID_PATH not in sys.path:
    sys.path.append(FASTREID_PATH)

import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

BASE_DIR = os.path.dirname(__file__)
from modules.v_e_local.download_models import ensure_vehicle_model_assets

ensure_vehicle_model_assets()

from ultralytics import YOLO
import torchvision.transforms as T

from database import get_active_visit_id, save_vehicle_entry_record

try:
    fastreid_config = importlib.import_module("fastreid.config")
    fastreid_engine = importlib.import_module("fastreid.engine")
    get_cfg = fastreid_config.get_cfg
    DefaultPredictor = fastreid_engine.DefaultPredictor
except Exception as exc:
    get_cfg = None
    DefaultPredictor = None
    print(f"Warning: FastReID is unavailable ({exc}); vehicle embeddings will be skipped")

logger = logging.getLogger(__name__)

DETECTION_MODEL_PATH = os.getenv("VEHICLE_DETECTION_MODEL", os.path.join(BASE_DIR, "yolov8n.pt"))
REID_WEIGHTS_PATH = os.getenv("VEHICLE_REID_WEIGHTS", os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth"))
REID_CONFIG_PATH = os.getenv(
    "VEHICLE_REID_CONFIG",
    os.path.join(FASTREID_PATH, "configs", "VehicleID", "bagtricks_R50-ibn.yml"),
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

if not os.path.isfile(DETECTION_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle detection model not found: {DETECTION_MODEL_PATH}")

detector = YOLO(DETECTION_MODEL_PATH)

cfg = None
predictor = None

if get_cfg is not None and DefaultPredictor is not None:
    cfg = get_cfg()
    if os.path.isfile(REID_CONFIG_PATH):
        cfg.merge_from_file(REID_CONFIG_PATH)
        cfg.MODEL.WEIGHTS = REID_WEIGHTS_PATH
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.isfile(cfg.MODEL.WEIGHTS):
            predictor = DefaultPredictor(cfg)
        else:
            logger.warning("FastReID checkpoint not found at %s; vehicle embeddings will be skipped", cfg.MODEL.WEIGHTS)
    else:
        logger.warning("FastReID config not found at %s; vehicle embeddings will be skipped", REID_CONFIG_PATH)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def get_vehicle_embedding(crop):
    if predictor is None or crop is None or crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb)
    tensor = tensor.unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        embedding = predictor.model({"images": tensor})

    embedding = embedding.cpu().numpy().flatten()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None

    return embedding / norm


def _resolve_visit_id(employee_id, visit_id):
    if visit_id:
        return visit_id
    if employee_id:
        return get_active_visit_id(employee_id)
    return None


def run_vehicle_entry_logic(
    vehicle_frame_queue, 
    entry_id_queue, 
    person_data_queue,
    ready_event=None, 
    # employee_id=None, 
    # visit_id=None, 
    camera_id=None
):
    print("VEHICLE WORKER STARTED")
    print("VEHICLE ENTRY QUEUE ID: ", id(person_data_queue))
    if ready_event is not None:
        ready_event.set()
        print("Vehicle worker ready")
    
    logger.info("Starting vehicle entry logic")
    # resolved_visit_id = _resolve_visit_id(employee_id, visit_id)
    
    tracked_vehicles = {}
    MIN_EMBEDDINGS_TO_SAVE = 3
    STALE_TRACK_FRAMES = 15.0
    
    # if not resolved_visit_id:
    #     logger.warning("No visit context available for vehicle entry; detected vehicles will not be saved to database.")
    
    logger.info("Waiting for person recognition context...")
    last_cleanup_time = time.time()
    
    current_employee_id = None
    current_visit_id = None
    vehicle_saved = False
    
    print("VEHICLE QUEUE OBJECT ", entry_id_queue)
    
    while True:
        try:
            while current_employee_id is None:
                try:
                    print("WAITING FOR PERSON DATA...")
                    print("QUEUE SIZE =", person_data_queue.qsize())
                    
                    data = person_data_queue.get(timeout=1)
                    print("RAW DATA RECEIVED =", data)
                    print("WAITING FOR PERSON DATA")
                    
                    current_employee_id = data.get("employee_id")
                    current_visit_id = data.get("visit_id")
                    
                    print(
                        f"RECEIVED PERSON CONTEXT -> "
                        f"emp={current_employee_id}, "
                        f"visit={current_visit_id}"
                    )
                except Exception as e:
                    print("NO PERSON DATA: ", e)
                    continue
        except Exception:
            pass
                
        frame = vehicle_frame_queue.get()
        print("Vehicle worker received frame")
        if frame is None: break
        
        results = detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        print("Boxes:", results.boxes)

        if results.boxes is not None:
            print("Detected classes:", results.boxes.cls)
        
                
        
        if results.boxes is not None and results.boxes.id is not None:
            box_ids = results.boxes.id.int().cpu().tolist()
            cls_indices = results.boxes.cls.int().cpu().tolist()
            
            
                        
            for idx, track_id in enumerate(box_ids):
                class_name = results.names[cls_indices[idx]]
                if class_name != "motorcycle": 
                    continue
                
                # if track_id in tracked_vehicles and tracked_vehicles[track_id]["saved"]:
                #     continue
                
                if track_id not in tracked_vehicles:
                    tracked_vehicles[track_id] = {"embeddings": [], "saved": False}
                    
                tracked_vehicles[track_id]["last_seen"] = time.time()
                
                
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx].cpu().tolist())
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                embedding = get_vehicle_embedding(crop)
                
                if embedding is not None:
                    tracked_vehicles[track_id]["embeddings"].append(embedding)
                    
                print("Current employee =", current_employee_id)
                print("Current visit =", current_visit_id)
                print("Embeddings =", len(tracked_vehicles[track_id]["embeddings"]))    
                    
                if len(tracked_vehicles[track_id]["embeddings"]) >= MIN_EMBEDDINGS_TO_SAVE and not vehicle_saved:
                    mean_embedding = np.mean(tracked_vehicles[track_id]["embeddings"], axis=0)
                    norm = np.linalg.norm(mean_embedding)
                    if norm == 0:
                        continue  
                                  
                    entry_id = None
                    
                    try:
                        if current_employee_id and current_visit_id:
                            print(
                                "Saving vehicle:",
                                current_employee_id,
                                current_visit_id,
                                track_id
                            )
                            
                            print("Calling save_vehicle_entry_record")
                            if not current_employee_id:
                                print("NO EMPLOYEE ID")
                                continue
                            
                            if not current_visit_id:
                                print("NO VISIT ID")
                                continue 
                            
                            print(
                                "ABOUT TO SAVE VEHICLE ENTRY RECORD:",
                                current_employee_id,
                                current_visit_id,
                                len(tracked_vehicles[track_id]["embeddings"]),
                            )
                            
                            entry_id = save_vehicle_entry_record(
                                employee_id=current_employee_id,
                                visit_id=current_visit_id,
                                vehicle_embedding=mean_embedding / norm if norm > 0 else None,
                                vehicle_class=results.names[cls_indices[idx]],
                                plate_number=None,
                                camera_id=camera_id,
                            )
                            print("Returned entry_id =", entry_id)
                        
                                                
                        if entry_id:
                            vehicle_saved = True
                            print("PUSHING ENTRY ID TO PLATE:", entry_id)
                            # tracked_vehicles[track_id]["saved"] = True
                            
                            
                            entry_id_queue.put(entry_id)
                            print(
                                "QUEUE OBJECT IN VEHICLE: ", id(entry_id_queue)
                            )
                            print("PUT COMPLETE:", entry_id)
                            logger.info(f"Saved vehicle entry record with ID: {entry_id}")
                            logger.info("Saved vehicle entry record for detected vehicle (class=%s)", results.names[cls_indices[idx]])
                    except Exception as exc:
                        logger.exception("Failed to save vehicle entry record: %s", exc)
                        
        current_time = time.time()
        
        
        if current_time - last_cleanup_time > 30:
            tracked_vehicles = {
                tid: data for tid, data in tracked_vehicles.items()
                if current_time - data["last_seen"] <= STALE_TRACK_FRAMES
            }
            last_cleanup_time = current_time