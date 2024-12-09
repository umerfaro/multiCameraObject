


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
import threading
import os
import logging

from config import TrackingConfig
from multicamtracker import MultiCameraTracker

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiCamAPI")

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
    global processing_done, tracker

    if not os.path.exists(video1_path):
        logger.error(f"video1_path not found: {video1_path}")
        processing_done = True
        return

    if not os.path.exists(video2_path):
        logger.error(f"video2_path not found: {video2_path}")
        processing_done = True
        return

    try:
        config = TrackingConfig(
            conf_threshold=conf,
            iou_threshold=iou_thresh,
            cos_threshold=cos_thresh
        )
    except Exception as e:
        logger.error(f"Failed to create TrackingConfig: {e}")
        processing_done = True
        return

    try:
        tracker = MultiCameraTracker(config)
    except Exception as e:
        logger.error(f"Failed to initialize MultiCameraTracker: {e}")
        processing_done = True
        return

    try:
        # Pass paths directly, let process_videos handle readers internally
        tracker.process_videos(video1_path, video2_path, output_path)
        processing_done = True
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        processing_done = True


@app.post("/start_tracking")
def start_tracking(req: TrackingRequest):
    global worker_thread, processing_done, tracker

    if not os.path.exists(req.video1):
        raise HTTPException(status_code=400, detail="video1 does not exist.")
    if not os.path.exists(req.video2):
        raise HTTPException(status_code=400, detail="video2 does not exist.")

    if worker_thread and worker_thread.is_alive():
        raise HTTPException(status_code=400, detail="Tracking is already in progress.")

    # Reset state
    processing_done = False
    tracker = None

    worker_thread = threading.Thread(
        target=run_tracker, 
        args=(req.video1, req.video2, req.output, req.conf, req.iou_thresh, req.cos_thresh),
        daemon=True
    )
    worker_thread.start()

    return {"message": "Tracking started"}


@app.get("/status")
def status():
    global worker_thread, processing_done, tracker
    if worker_thread is None:
        return {"status": "not started"}
    elif worker_thread.is_alive():
        return {"status": "processing"}
    else:
        # Completed processing, tracker may be None if initialization failed
        if tracker:
            return {"status": "done", "total_visits": tracker.total_visit_count}
        else:
            return {"status": "failed", "message": "Tracker did not complete successfully"}


if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

