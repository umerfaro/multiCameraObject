

import argparse
from config import TrackingConfig
from frame_reader import FrameReader
from multicamtracker import MultiCameraTracker

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

    # Initialize readers
    from queue import Queue
    queue1 = Queue(maxsize=20)
    queue2 = Queue(maxsize=20)

    # Validate input videos
    import os
    if not os.path.exists(args.video1):
        raise FileNotFoundError(f"video1 not found at {args.video1}")
    if not os.path.exists(args.video2):
        raise FileNotFoundError(f"video2 not found at {args.video2}")

    reader1 = FrameReader(args.video1, queue1, name='Camera1')
    reader2 = FrameReader(args.video2, queue2, name='Camera2')

    try:
        tracker = MultiCameraTracker(config)
    except Exception as e:
        print(f"Failed to initialize tracker: {e}")
        return

    try:
        tracker.process_videos(reader1, reader2, args.output)
    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        # Ensure resources are released
        reader1.stop()
        reader2.stop()


if __name__ == "__main__":
    main()

