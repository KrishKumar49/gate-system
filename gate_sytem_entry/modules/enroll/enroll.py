import os
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import requests

from insightface.app import FaceAnalysis

# Ensures this file can always find database.py even if moved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from database import save_employee_embedding

face_app = None

MODEL_HOME = os.path.join(os.path.dirname(__file__), "insightface_models")
MODEL_ROOT = os.path.join(MODEL_HOME, "models")
MODEL_CANDIDATES = (
    "buffalo_m",
    "buffalo_l",
)
MODEL_PACK_URLS = {
    "buffalo_m": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
    "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}


def _model_pack_path(model_name):
    return os.path.join(MODEL_ROOT, model_name)


def _has_model_files(model_name):
    model_path = _model_pack_path(model_name)

    return os.path.isdir(model_path) and any(
        file_name.endswith(".onnx")
        for file_name in os.listdir(model_path)
    )


def _download_model_pack(model_name):
    url = MODEL_PACK_URLS[model_name]
    os.makedirs(MODEL_ROOT, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
        temp_zip_path = temp_zip.name

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(temp_zip_path, "wb") as zip_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    zip_file.write(chunk)

        with zipfile.ZipFile(temp_zip_path) as archive:
            archive.extractall(path=MODEL_ROOT)
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)


def _download_video(video_url, destination_path):
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return False

    with open(destination_path, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                video_file.write(chunk)

    return True


def _ensure_model_pack(model_name):
    if _has_model_files(model_name):
        return

    _download_model_pack(model_name)

    if not _has_model_files(model_name):
        raise RuntimeError(f"Downloaded InsightFace model pack '{model_name}' is incomplete")


def get_face_app():
    global face_app

    if face_app is None:
        last_error = None

        for model_name in MODEL_CANDIDATES:
            try:
                _ensure_model_pack(model_name)

                face_app = FaceAnalysis(
                    name=model_name,
                    root=MODEL_HOME,
                    providers=['CPUExecutionProvider']
                )
                face_app.prepare(
                    ctx_id=-1,
                    det_size=(640, 640)
                )
                break
            except Exception as error:
                last_error = error
                face_app = None

        if face_app is None:
            raise RuntimeError(f"Could not initialize InsightFace models: {last_error}")

    return face_app

def enroll_employee(video_url, employee_id):
    os.makedirs("temp", exist_ok=True)

    temp_video_path = f"temp/{employee_id}.mp4"

    try:
        model = get_face_app()
    except Exception as error:
        return {
            "status": "failed",
            "message": f"Could not initialize face model: {error}"
        }

    if not _download_video(video_url, temp_video_path):
        return {
            "status": "failed",
            "message": "Could not download video"
        }

    video = None

    try:
        video = cv2.VideoCapture(temp_video_path)

        if not video.isOpened():
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            return {
                "status": "failed",
                "message": "Could not open video"
            }

        collected_face_data = []

        frame_count = 0


        # output_faces_dir = f'faces/{employee_id}'

        # os.makedirs(
        #     output_faces_dir,
        #     exist_ok=True
        # )

        os.makedirs(
            'embeddings',
            exist_ok=True
        )

        while True:

            ret, frame = video.read()

            if not ret:
                break

            current_frame = frame_count
            frame_count += 1

            if current_frame % 15 != 0:
                continue

            x1, y1, x2, y2 = 0, 0, 0, 0 


            faces = model.get(frame)

            if len(faces) > 0:
                face = max(
                    faces,
                    key=lambda f:
                    (f.bbox[2] - f.bbox[0]) *
                    (f.bbox[3] - f.bbox[1])
                )

                x1, y1, x2, y2 = map(
                    int,
                    face.bbox
                )

                h, w, _ = frame.shape

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                face_width = x2 - x1
                face_height = y2 - y1

                face_crop = None
                blur_score = 0.0
                eye_ratio = 0.0
                embedding = None
                

                if face.det_score < 0.7: 
                    pass
                elif face_width < 80 or face_height < 80: 
                    pass
                else:
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        pass
                    else:
                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                        blur_score = cv2.Laplacian(
                            gray_face,
                            cv2.CV_64F
                        ).var()

                        print(
                            f"Frame {current_frame}: Blur Score: {blur_score:.2f}",
                            flush=True
                        )

                        if blur_score < 50:
                            continue

                        left_eye = face.kps[0]
                        right_eye = face.kps[1]
                        eye_distance = abs(right_eye[0] - left_eye[0])

                        if face_width > 0: 
                            eye_ratio = eye_distance / face_width
                        

                        if eye_ratio < 0.25: 
                            pass
                        else:
                            embedding = face.normed_embedding
                            is_duplicate = False
                            for saved_data in collected_face_data:
                                saved_embedding = saved_data[0] 
                                similarity = np.dot(embedding, saved_embedding)
                                if similarity > 0.97: 
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                collected_face_data.append((embedding, face.det_score, blur_score, eye_ratio, current_frame))

                                # cv2.imwrite(f"{output_faces_dir}/{current_frame}.jpg", face_crop)

            


        if len(collected_face_data) < 5:
            return {
                "status": "failed",
                "message": "Not enough quality samples"
            }


        final_embedding = np.mean(
            [data[0] for data in collected_face_data],
            axis=0
        )

        final_embedding = final_embedding / np.linalg.norm(final_embedding)

        all_embeddings = np.array(
            [data[0] for data in collected_face_data]
        )

        np.save(
            f'embeddings/{employee_id}_all.npy',
            all_embeddings
        )

        np.save(
            f'embeddings/{employee_id}_mean.npy',
            final_embedding
        )
        
        # mean_path = f'embeddings/{employee_id}_mean.npy'
        # all_path = f'embeddings/{employee_id}_all.npy'

        # save_employee_embedding(
        #     employee_id,
        #     mean_path,
        #     all_path,
        #     len(collected_face_data)
        # )

        save_employee_embedding(
            employee_id,
            final_embedding,
            all_embeddings,
            len(collected_face_data)
        )


        return {
            "status": "success",
            "employeeId": employee_id,
            "embeddingsCount": len(collected_face_data),
            "embeddingStored": True
            
            # "savedEmbeddings": {
            #     "mean": f"embeddings/{employee_id}_mean.npy",
            #     "all": f"embeddings/{employee_id}_all.npy"
            # }
            
            
            
            # To also return all individual embeddings used to compute the final
            # embedding, uncomment the following line. This will include the
            # raw per-sample embeddings (as lists) in the response.
            # "all_embeddings": [data[0].tolist() for data in collected_face_data],
        }

    except Exception as e:
        return {
            "status": "failed",
            "message": str(e)
        }
    finally:
        if video is not None:
            video.release()

        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
            
            
            