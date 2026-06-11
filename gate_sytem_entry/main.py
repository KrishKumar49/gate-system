import multiprocessing 
import cv2

from modules.v_e_local.vehicle_entry import run_vehicle_entry_logic
from modules.enroll.recognize import run_person_logic
from modules.enroll.plate_recognition import run_plate_logic

if __name__ == "__main__":
    
    person_data_queue = multiprocessing.Queue(maxsize=5)
    entry_id_queue = multiprocessing.Queue(maxsize=10)
    
    person_frame_queue = multiprocessing.Queue(10)
    vehicle_frame_queue = multiprocessing.Queue(10)
    plate_frame_queue = multiprocessing.Queue(10)
    
    p0 = multiprocessing.Process(target=run_person_logic, args=(person_frame_queue, person_data_queue))
    
    p1 = multiprocessing.Process(target=run_vehicle_entry_logic,
                                  args=(vehicle_frame_queue, entry_id_queue, person_data_queue))  
    
    p2 = multiprocessing.Process(target=run_plate_logic, args=(plate_frame_queue, entry_id_queue))
    
    p0.start()
    p1.start()
    p2.start()
    
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            for q in [person_frame_queue, vehicle_frame_queue, plate_frame_queue]:
                if not q.full():
                    q.put(frame)
            
            # cv2.imshow("Live Feed", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        cap.release()
        for q in [person_frame_queue, vehicle_frame_queue, plate_frame_queue]:
            q.put(None)  # Signal workers to exit
        entry_id_queue.put(None)  # Signal workers to exit
        person_data_queue.put(None)  # Signal workers to exit
        # cv2.destroyAllWindows()
        
        p0.terminate()
        p1.terminate()
        p2.terminate()