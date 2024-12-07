# Multi-Camera Tracking System

This repository contains a multi-camera person tracking system using YOLOv8 for detection and OSNet for person Re-ID. It processes two input video streams, tracks individuals across both cameras, and can optionally save the combined visualization to an output video file. The system uses a separate visualization thread to display real-time tracking results and is exposed as a FastAPI-based HTTP API.

## Features

- **Multi-camera tracking:** Tracks persons in two video streams simultaneously.
- **YOLOv8 detection:** Uses a lightweight YOLOv8 model for efficient person detection.
- **OSNet-based Re-ID:** Re-identifies individuals across cameras using deep Re-ID features.
- **Cross-camera matching:** Assigns global IDs to individuals to track them across both cameras.
- **Visualization thread:** Displays tracking results in real-time in a separate window.
- **HTTP API interface:** Start tracking and monitor progress via a RESTful API.

## Directory Structure


- **config.py:** Contains the `TrackingConfig` dataclass to configure thresholds and model paths.
- **track.py:** Defines the `Track` class that stores track details and Re-ID features.
- **frame_reader.py:** Implements `FrameReader` for asynchronous video frame reading.
- **multi_camera_tracker.py:** Implements the `MultiCameraTracker` class that coordinates detection, feature extraction, tracking, cross-camera matching, and visualization.
- **main.py:** (Optional) CLI entry point for running the tracker without the API.
- **app.py:** FastAPI application exposing endpoints to start tracking and monitor the status.
- **weights/:** Directory to store the Re-ID weights (`osnet_x0_25.pth.tar`) and YOLO model (`yolov8n.pt`).

## Requirements

Install dependencies via:
```bash
pip install -r requirements.txt


start the API server:

```bash
python app.py


curl -X POST http://localhost:8000/start_tracking \
     -H "Content-Type: application/json" \
     -d '{
          "video1": "cam11.mp4",
          "video2": "cam44.mp4",
          "output": "output.avi",
          "conf": 0.3,
          "iou_thresh": 0.3,
          "cos_thresh": 0.8
         }'


Parameters:

video1, video2: Paths to input video files.
output (optional): Path to save the output visualization video.
conf: YOLO confidence threshold.
iou_thresh: IOU threshold for tracking.
cos_thresh: Cosine similarity threshold for Re-ID matching.