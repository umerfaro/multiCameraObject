import argparse
from multicamtracker import MultiCameraTracker
from config import TrackingConfig

def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-Camera Tracking System")
    parser.add_argument("--video1", type=str, required=True, help="Path to first video file")
    parser.add_argument("--video2", type=str, required=True, help="Path to second video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IOU threshold for tracker")
    parser.add_argument("--cos-thresh", type=float, default=0.8, help="Cosine similarity threshold for ReID")
    
    args = parser.parse_args()
    
    config = TrackingConfig(
        conf_threshold=args.conf,
        iou_threshold=args.iou_thresh,
        cos_threshold=args.cos_thresh
    )
    tracker = MultiCameraTracker(config)
    tracker.process_videos(args.video1, args.video2, args.output)   

if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python run.py --video1 cam11.mp4 --video2 cam44.mp4 --output output.avi