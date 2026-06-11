import multiprocessing
import cv2

from modules.v_e_local.vehicle_entry import run_vehicle_entry_logic
from modules.enroll.recognize import run_person_logic
from modules.enroll.plate_recognition import run_plate_logic


def start_gate_monitoring():

    person_data_queue = multiprocessing.Queue(5)
    entry_id_queue = multiprocessing.Queue(10)

    person_frame_queue = multiprocessing.Queue(10)
    vehicle_frame_queue = multiprocessing.Queue(10)
    plate_frame_queue = multiprocessing.Queue(10)

    p0 = multiprocessing.Process(
        target=run_person_logic,
        args=(person_frame_queue, person_data_queue)
    )

    p1 = multiprocessing.Process(
        target=run_vehicle_entry_logic,
        args=(vehicle_frame_queue,
              entry_id_queue,
              person_data_queue)
    )

    p2 = multiprocessing.Process(
        target=run_plate_logic,
        args=(plate_frame_queue,
              entry_id_queue)
    )

    p0.start()
    p1.start()
    p2.start()

    cap = cv2.VideoCapture(0)

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            for q in [
                person_frame_queue,
                vehicle_frame_queue,
                plate_frame_queue
            ]:
                if not q.full():
                    q.put(frame)

    finally:

        cap.release()

        for q in [
            person_frame_queue,
            vehicle_frame_queue,
            plate_frame_queue
        ]:
            q.put(None)

        p0.join(timeout=5)
        p1.join(timeout=5)
        p2.join(timeout=5)
        

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    start_gate_monitoring()