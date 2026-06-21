from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import multiprocessing

from modules.enroll.enroll import enroll_employee
from monitor import start_gate_monitoring
from database import delete_employee

from monitor_exit import start_exit_monitoring as run_exit_monitoring

exit_monitor_process = multiprocessing.Process(target=run_exit_monitoring)

app = FastAPI()

class EnrollRequest(BaseModel):
    employeeId: str
    videoUrl: str


monitor_process = None
exit_monitor_process = None

@app.post("/start_monitoring")
def start_monitoring():
    global monitor_process
    if monitor_process is None or not monitor_process.is_alive():
        monitor_process = multiprocessing.Process(target=start_gate_monitoring)
        monitor_process.start()
    return {"status": "Monitoring started"}


@app.post("/stop_monitoring")
def stop_monitoring():
    global monitor_process
    if monitor_process is not None and monitor_process.is_alive():
        monitor_process.terminate()
        monitor_process = None
    return {"status": "Monitoring stopped"}


@app.get("/status")
def get_status():
    global monitor_process
    if monitor_process is not None and monitor_process.is_alive():
        return {"status": "Monitoring is running"}
    else:
        return {"status": "Monitoring is stopped"}
    

@app.get("/health") 
def get_health():
    return {"status": "Service is running"}


@app.post("/enroll_person")
def enroll(data: EnrollRequest):

    result = enroll_employee(
        data.videoUrl,
        data.employeeId
    )

    return result

@app.delete("/employee/{employee_id}")
def remove_employee(employee_id: str):
    return delete_employee(employee_id)

@app.post("/start_exit_monitoring")
def start_exit_monitoring():
    global exit_monitor_process
    if exit_monitor_process is None or not exit_monitor_process.is_alive():
        exit_monitor_process = multiprocessing.Process(target=run_exit_monitoring)
        exit_monitor_process.start()
    return {"status": "Exit monitoring started"}


@app.post("/stop_exit_monitoring")
def stop_exit_monitoring():
    global exit_monitor_process
    if exit_monitor_process is not None and exit_monitor_process.is_alive():
        exit_monitor_process.terminate()
        exit_monitor_process = None
    return {"status": "Exit monitoring stopped"}


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)