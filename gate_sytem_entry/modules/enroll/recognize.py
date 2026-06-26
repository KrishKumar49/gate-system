import os
import tempfile

import cv2
import numpy as np
import requests

import sys
import os

from collections import Counter, deque

import logging
# This line finds the path to your project root (gate_sytem_entry/) 
# and adds it to Python's system path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now, absolute imports work!
from database import get_all_employee_embeddings
from modules.enroll.enroll import get_face_app
from database import (
    get_all_employee_embeddings,
    save_recognition_event,
    get_active_visit_id,
    create_visit_record
)
from modules.enroll.enroll import get_face_app

MATCH_THRESHOLD = 0.60
FRAME_SKIP = 1
RECOGNITION_COOLDOWN = 3  # seconds



def _load_known_embeddings():
    database_rows = get_all_employee_embeddings()

    known_embeddings = {}

    for employee_id, embedding in database_rows:
        arr = (
            np.array(embedding)
            if isinstance(embedding, list)
            else np.array(eval(embedding))
        )

        known_embeddings.setdefault(
            employee_id,
            []
        ).append(arr)

    return known_embeddings


def run_person_logic(
    frame_queue,
    person_data_queue,
    ready_event=None
):
    print("Loading embeddings from database")

    known_embeddings = _load_known_embeddings()

    print(
        "Known employees:",
        known_embeddings.keys()
    )
    logging.info(
        f"PERSON PROCESS {os.getpid()} - QUEUE {id(person_data_queue)}"
    )

    for emp, embs in known_embeddings.items():
        print(emp, len(embs))

    model = get_face_app()

    if ready_event is not None:
        print("Person worker ready")
        ready_event.set()

    frame_count = 0
    last_seen = {}
    active_people = {}
    
    recognition_votes = Counter()
    recognition_window = deque()
    VOTE_THRESHOLD = 2
    WINDOW_SIZE = 10

    print("Person recognition worker started")

    while True:

        frame = frame_queue.get()

        print("Person worker received frame")

        if frame is None:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        faces = model.get(frame)

        print(
            f"Faces detected: {len(faces)}"
        )

        for face in faces:

            embedding = face.normed_embedding

            best_match_id = None
            best_match_score = -1.0

            for (
                employee_id,
                embs
            ) in known_embeddings.items():

                for stored_emb in embs:

                    similarity = np.dot(
                        embedding,
                        stored_emb
                    )

                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_id = employee_id

            print(
                f"Best match = {best_match_id}, "
                f"Score = {best_match_score:.4f}, "
                f"Threshold = {MATCH_THRESHOLD}"
            )

            if best_match_score <= MATCH_THRESHOLD:
                continue
            
            
            
            recognition_window.append(best_match_id)
            recognition_votes[best_match_id] += 1

            if len(recognition_window) > WINDOW_SIZE:
                old = recognition_window.popleft()
                recognition_votes[old] -= 1

                if recognition_votes[old] <= 0:
                    del recognition_votes[old]

            if recognition_votes[best_match_id] < VOTE_THRESHOLD:
                continue

            current_time = (
                cv2.getTickCount()
                / cv2.getTickFrequency()
            )
            
            print(
                "NOW =", current_time,
                "LAST SEEN =", last_seen.get(best_match_id, None)
            )

            if (
                best_match_id in last_seen
                and
                current_time
                - last_seen[best_match_id]
                < RECOGNITION_COOLDOWN
            ):
                print(
                    f"Employee {best_match_id} "
                    f"already logged recently"
                )
                continue

            last_seen[best_match_id] = current_time

            x1, y1, x2, y2 = map(
                int,
                face.bbox
            )
            
            width = x2 - x1
            height = y2 - y1
            
            print(width, height)

            print(
                f"Recognized employee "
                f"{best_match_id} "
                f"with score "
                f"{best_match_score:.4f}"
            )

            save_recognition_event(
                employee_id=best_match_id,
                confidence=float(
                    best_match_score
                ),
                camera_type="ENTRY"
            )

            visit_id = get_active_visit_id(best_match_id)
            print("ACTIVE VISIT: ", visit_id)

            if visit_id is None:
                visit_id = create_visit_record(best_match_id)
            
            print("FINAL VISIT: ", visit_id)
                
            print(
                f"SENDING PERSON CONTEXT -> "
                f"emp={best_match_id}, "
                f"visit={visit_id}"
            )
            
            print("PERSON QUEUE ID =", id(person_data_queue))
            print("PUTTING INTO PERSON QUEUE")

            person_data_queue.put({
                "employee_id": best_match_id,
                "visit_id": visit_id
            })
            
            active_people[best_match_id] = True
            
            print("PUT COMPLETED")
            print("QUEUE SIZE AFTER PUT =", person_data_queue.qsize())

            recognition_votes.clear()
            recognition_window.clear()
                











# def start_live_recognition(camera_source=0):
#     print("loading embedding from database")
#     known_embeddings = _load_known_embeddings()

#     print("starting video capture")
#     total_embeddings = sum(len(employee_embeddings) for employee_embeddings in known_embeddings.values())
#     print(f"loaded {len(known_embeddings)} employees with {total_embeddings} embeddings from database")
    
#     model = get_face_app()
    
#     video = cv2.VideoCapture(camera_source)
    
#     if not video.isOpened():
#         print("Could not open video source")
#         return
    
#     frame_count = 0
    
#     last_seen = {}
    
#     while True:
#         ret, frame = video.read()
        
#         if not ret:
#             print("Could not read frame from video source")
#             break
        
#         frame_count += 1
        
#         if frame_count % FRAME_SKIP != 0:
#             continue
        
#         faces = model.get(frame)
        
#         for face in faces:
#             embedding = face.normed_embedding
            
#             best_match_id = None
#             best_match_score = -1.0
            
#             for employee_id, employee_embeddings in known_embeddings.items():
#                 for stored_embedding in employee_embeddings:
#                     # i think we have to convert stored_embedding to numpy array here as they are in strings
#                     similarity = np.dot(embedding, stored_embedding)

#                     if similarity > best_match_score:
#                         best_match_score = similarity
#                         best_match_id = employee_id
            
#             if best_match_score > MATCH_THRESHOLD:
                
#                 if best_match_id in last_seen and (cv2.getTickCount() - last_seen[best_match_id]) / cv2.getTickFrequency() < 1:
#                     print(f"Employee {best_match_id} seen again with score {best_match_score}")
#                     continue
                
                
#                 last_seen[best_match_id] = cv2.getTickCount()
                
#                 print(f"Recognized employee {best_match_id} with score {best_match_score}")
                
#                 x1, y1, x2, y2 = map(int, face.bbox)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f"{best_match_id} ({best_match_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
#         cv2.imshow("Live Recognition", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#         # Remove employees not seen for a while
#         current_time = cv2.getTickCount()
#         to_remove = []
        
#         for employee_id, last_seen_time in last_seen.items():
#             if (current_time - last_seen_time) / cv2.getTickFrequency() > 5:  # 5 seconds timeout
#                 to_remove.append(employee_id)
        
#         for employee_id in to_remove:
#             del last_seen[employee_id]
            
#     video.release()
#     cv2.destroyAllWindows()