from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
import threading
import os

from config import TrackingConfig
from multicamtracker import MultiCameraTracker

app = FastAPI()

class TrackingRequest(BaseModel):
    video1: str
    video2: str
    output: Optional[str] = None
    conf: float = 0.3
    iou_thresh: float = 0.3
    cos_thresh: float = 0.8

# Global tracker and worker thread references
tracker = None
worker_thread = None
processing_done = False

def run_tracker(video1_path, video2_path, output_path, conf, iou_thresh, cos_thresh):
    global processing_done
    config = TrackingConfig(
        conf_threshold=conf,
        iou_threshold=iou_thresh,
        cos_threshold=cos_thresh
    )
    global tracker
    tracker = MultiCameraTracker(config)
    tracker.process_videos(video1_path, video2_path, output_path)
    processing_done = True

@app.post("/start_tracking")
def start_tracking(req: TrackingRequest):
    global worker_thread, processing_done

    if not os.path.exists(req.video1):
        raise HTTPException(status_code=400, detail="video1 does not exist.")
    if not os.path.exists(req.video2):
        raise HTTPException(status_code=400, detail="video2 does not exist.")

    if worker_thread and worker_thread.is_alive():
        raise HTTPException(status_code=400, detail="Tracking is already in progress.")

    processing_done = False
    worker_thread = threading.Thread(
        target=run_tracker, 
        args=(req.video1, req.video2, req.output, req.conf, req.iou_thresh, req.cos_thresh),
        daemon=True
    )
    worker_thread.start()
    return {"message": "Tracking started"}

@app.get("/status")
def status():
    global worker_thread, processing_done
    if worker_thread is None:
        return {"status": "not started"}
    elif worker_thread.is_alive():
        return {"status": "processing"}
    else:
        # Completed processing
        return {"status": "done", "total_visits": tracker.total_visit_count if tracker else None}

if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
