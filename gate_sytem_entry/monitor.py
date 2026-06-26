import multiprocessing
import cv2
import time

import os

import logging

from modules.v_e_local.vehicle_entry import run_vehicle_entry_logic
from modules.enroll.recognize import run_person_logic
from modules.enroll.plate_recognition import run_plate_logic


def start_gate_monitoring():
    print("START METHOD: ", multiprocessing.get_all_start_methods())

    person_data_queue = multiprocessing.Queue(5)
    logging.info(
        f"MAIN PROCESS {os.getpid()} - QUEUE {id(person_data_queue)}"
    )
    # entry_id_queue = multiprocessing.Queue(10)
    plate_entry_queue = multiprocessing.Queue(10)
    
    # plate_entry_queue.put("TEST123")

    person_frame_queue = multiprocessing.Queue(10)
    vehicle_frame_queue = multiprocessing.Queue(10)
    plate_frame_queue = multiprocessing.Queue(10)

    # readiness events: workers set these once their models/assets are loaded
    person_ready = multiprocessing.Event()
    vehicle_ready = multiprocessing.Event()
    plate_ready = multiprocessing.Event()

    p0 = multiprocessing.Process(
        target=run_person_logic,
        args=(person_frame_queue, person_data_queue, person_ready)
    )

    p1 = multiprocessing.Process(
        target=run_vehicle_entry_logic,
        args=(
            vehicle_frame_queue,
            plate_entry_queue,
            person_data_queue,
            vehicle_ready
            )
    )

    p2 = multiprocessing.Process(
        target=run_plate_logic,
        args=(plate_frame_queue,
              plate_entry_queue,
              plate_ready)
    )

    p0.start()
    p1.start()
    p2.start()
    # wait for workers to signal readiness (with a timeout)
    all_ready = person_ready.wait(timeout=120) and vehicle_ready.wait(timeout=120) and plate_ready.wait(timeout=120)
    if not all_ready:
        print("Warning: not all workers signalled ready within timeout")

    cap = cv2.VideoCapture("https://ik.imagekit.io/6f8hdxg1w/WhatsApp%20Video%202026-06-21%20at%2011.34.52%20PM.mp4")
    print("Video opened:", cap.isOpened())
    try:
        while True:

            ret, frame = cap.read()

            print("ret =", ret)

            if not ret:
                print("End of video or failed to read frame")
                break

            for q in [
                person_frame_queue,
                vehicle_frame_queue,
                plate_frame_queue
            ]:
                if not q.full():
                    # throttle frame sending so workers don't get overwhelmed
                    q.put(frame)

            # small sleep to avoid reading the entire video too fast
            time.sleep(0.03)

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