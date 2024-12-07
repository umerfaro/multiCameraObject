import torch
import numpy as np
import cv2
import torchreid
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from scipy.optimize import linear_sum_assignment
import time
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    conf_threshold: float = 0.30
    cos_threshold: float = 0.80
    reid_model_name: str = "osnet_x1_0"
    reid_weights_path: str = "./weights/osnet_x1_0.pth.tar"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (640, 640)  # YOLOv5 input size
    batch_size: int = 2  # Process both frames together

class MultiCameraTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.detector = self._initialize_detector()
        self.reid_extractor = self._initialize_reid()
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")
        self.last_time = time.time()
        
    def _initialize_detector(self) -> torch.nn.Module:
        detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
        detector.agnostic = True
        detector.classes = [0]
        detector.conf = self.config.conf_threshold
        detector.to(self.config.device)
        for param in detector.parameters():
            param.requires_grad = False  # Disable gradient computation
        return detector

    def _initialize_reid(self) -> torchreid.utils.FeatureExtractor:
        extractor = torchreid.utils.FeatureExtractor(
            model_name=self.config.reid_model_name,
            model_path=self.config.reid_weights_path,
            device=self.config.device
        )
        extractor.model.eval()  # Ensure model is in eval mode
        return extractor

    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Efficiently preprocess frame for detection"""
        return cv2.resize(frame, target_size)

    def extract_features_batch(self, frames: List[np.ndarray], detections_list: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features for all detections in a batch"""
        all_features = []
        
        for frame, detections in zip(frames, detections_list):
            if len(detections) == 0:
                all_features.append(np.array([]))
                continue

            crops = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Preprocessing can be done here if needed
                crops.append(crop)

            if not crops:
                all_features.append(np.array([]))
                continue

            # Process all crops for one frame together
            with torch.no_grad(), autocast(enabled=True):
                try:
                    features = self.reid_extractor(crops)
                    features = features.cpu().numpy()
                    # Normalize features
                    features = features / np.linalg.norm(features, axis=1, keepdims=True)
                    all_features.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features: {e}")
                    all_features.append(np.array([]))

        return all_features

    def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
        video1 = cv2.VideoCapture(video1_path)
        video2 = cv2.VideoCapture(video2_path)
        video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

        num_frames = min(int(video1.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(video2.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        video_writer = None
        frames_processed = 0
        total_time = 0

        try:
            while frames_processed < num_frames:
                start_time = time.time()
                
                # Read frames
                ret1, frame1 = video1.read()
                ret2, frame2 = video2.read()
                if not ret1 or not ret2:
                    break

                # Convert to RGB and resize for detection
                frames_rgb = [
                    cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                ]

                # Run detection on batch
                with torch.no_grad():
                    detections = self.detector(frames_rgb)

                # Process detections
                dets1 = detections.xyxy[0].cpu().numpy()
                dets2 = detections.xyxy[1].cpu().numpy()

                # Extract ReID features in batch
                features_list = self.extract_features_batch(frames_rgb, [dets1, dets2])
                
                # Match detections
                matches = []
                if len(features_list[0]) > 0 and len(features_list[1]) > 0:
                    sim_matrix = features_list[0] @ features_list[1].T
                    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
                    
                    for r, c in zip(row_ind, col_ind):
                        if sim_matrix[r, c] >= self.config.cos_threshold:
                            matches.append((r, c))

                # Visualize results
                vis = self.visualize(frame1, frame2, dets1, dets2, matches)

                if output_path and video_writer is None:
                    h, w = vis.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, (w, h), True
                    )

                if video_writer is not None:
                    video_writer.write(vis)

                cv2.namedWindow("Multi-Camera Tracking", cv2.WINDOW_NORMAL)
                
                # Calculate and display FPS
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Multi-Camera Tracking", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frames_processed += 1
                total_time += (end_time - start_time)

                if frames_processed % 30 == 0:
                    avg_fps = frames_processed / total_time
                    logger.info(f"Average FPS: {avg_fps:.1f}")

        finally:
            video1.release()
            video2.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

    def visualize(self, frame1: np.ndarray, frame2: np.ndarray, 
                 dets1: np.ndarray, dets2: np.ndarray, 
                 matches: List[Tuple[int, int]]) -> np.ndarray:
        """Optimized visualization method"""
        vis1, vis2 = frame1.copy(), frame2.copy()
        
        for idx, (match1, match2) in enumerate(matches):
            color = tuple(map(int, self.colors[idx % len(self.colors)]))
            
            # Draw first frame
            box1 = dets1[match1][:4].astype(np.int32)
            cv2.rectangle(vis1, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
            cv2.putText(vis1, str(idx), (box1[0], box1[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw second frame
            box2 = dets2[match2][:4].astype(np.int32)
            cv2.rectangle(vis2, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)
            cv2.putText(vis2, str(idx), (box2[0], box2[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return np.hstack([vis1, vis2])

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking System")
    parser.add_argument("--video1", type=str, required=True)
    parser.add_argument("--video2", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--cos-thresh", type=float, default=0.8)
    
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




    # python multicam_tracking.py --video1 cam11.mp4 --video2 cam44.mp4 --output output.avi