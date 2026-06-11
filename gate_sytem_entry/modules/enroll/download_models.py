import os
import subprocess
import sys
import importlib

BASE_DIR = os.path.dirname(__file__)
MODEL_URL = os.getenv("VEHICLE_REID_MODEL_URL")
MODEL_PATH = os.getenv("VEHICLE_REID_WEIGHTS", os.path.join(BASE_DIR, "vehicleid_bot_R50-ibn.pth"))
FASTREID_PATH = os.getenv("FASTREID_PATH", os.path.join(BASE_DIR, "fast-reid"))
FASTREID_REPO_URL = os.getenv("FASTREID_REPO_URL", "https://github.com/JDAI-CV/fast-reid.git")
YOLO_MODEL_URL = os.getenv("VEHICLE_DETECTION_MODEL_URL")
YOLO_MODEL_PATH = os.getenv("VEHICLE_DETECTION_MODEL", os.path.join(BASE_DIR, "yolov8n.pt"))


def _path_has_content(path):
    return os.path.isdir(path) and any(os.scandir(path))


def ensure_vehicle_reid_model():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    if not MODEL_URL:
        raise RuntimeError(
            "VEHICLE_REID_MODEL_URL is not set. Upload the .pth file to Google Drive and set its share URL."
        )

    gdown = importlib.import_module("gdown")

    print(f"Downloading Vehicle ReID model to {MODEL_PATH}...")
    gdown.download(
        MODEL_URL,
        MODEL_PATH,
        quiet=False,
    )

    return MODEL_PATH


def ensure_fastreid_repo():
    if _path_has_content(FASTREID_PATH):
        return FASTREID_PATH

    if os.path.exists(FASTREID_PATH) and not _path_has_content(FASTREID_PATH):
        print(f"Removing empty FastReID directory at {FASTREID_PATH}")
        try:
            os.rmdir(FASTREID_PATH)
        except OSError:
            pass

    print(f"Cloning FastReID into {FASTREID_PATH}...")
    subprocess.check_call([
        "git",
        "clone",
        "--depth",
        "1",
        FASTREID_REPO_URL,
        FASTREID_PATH,
    ])

    return FASTREID_PATH


def ensure_yolo_detector_model():
    if os.path.exists(YOLO_MODEL_PATH):
        return YOLO_MODEL_PATH

    if not YOLO_MODEL_URL:
        raise RuntimeError(
            "VEHICLE_DETECTION_MODEL_URL is not set. Upload the YOLO file to Google Drive and set its share URL."
        )

    gdown = importlib.import_module("gdown")

    print(f"Downloading YOLO detector model to {YOLO_MODEL_PATH}...")
    gdown.download(
        YOLO_MODEL_URL,
        YOLO_MODEL_PATH,
        quiet=False,
    )
    return YOLO_MODEL_PATH


def ensure_vehicle_model_assets():
    ensure_vehicle_reid_model()
    ensure_fastreid_repo()
    ensure_yolo_detector_model()


if __name__ == "__main__":
    ensure_vehicle_model_assets()
    print("Model assets ready")