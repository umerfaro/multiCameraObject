import torch
import numpy as np
import cv2
import torchreid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from scipy.optimize import linear_sum_assignment
import time
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    conf_threshold: float = 0.30
    cos_threshold: float = 0.80
    reid_model_name: str = "osnet_x0_25"  # Changed to a lighter model
    reid_weights_path: str = "./weights/osnet_x0_25.pth.tar"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 2
    max_feature_distance: float = 0.7
    reid_keep_frames: int = 20  # Number of frames to keep ReID features

class Track:
    def __init__(self, feature, detection, track_id):
        self.id = track_id
        self.features = [feature]
        self.last_detection = detection
        self.missed_frames = 0
        self.consecutive_hits = 1
        self.last_position = detection[:4]

    def update(self, feature, detection):
        self.features.append(feature)
        if len(self.features) > 20:  # Keep only recent features
            self.features.pop(0)
        self.last_detection = detection
        self.missed_frames = 0
        self.consecutive_hits += 1
        self.last_position = detection[:4]

    def get_average_feature(self):
        return np.mean(self.features, axis=0)

class FrameReader(threading.Thread):
    """Asynchronous frame reader for video streams."""
    def __init__(self, video_path, queue, name=''):
        super().__init__()
        self.video = cv2.VideoCapture(video_path)
        self.queue = queue
        self.stopped = False
        self.name = name

    def run(self):
        while not self.stopped:
            ret, frame = self.video.read()
            if not ret:
                self.stopped = True
                self.queue.put(None)
                break
            self.queue.put(frame)

    def stop(self):
        self.stopped = True
        self.video.release()

class MultiCameraTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.detector = self._initialize_detector()
        self.reid_extractor = self._initialize_reid()
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")  # Increased color palette
        self.last_time = time.time()
        
        # Track management
        self.next_track_id = 0
        self.tracks_cam1 = {}
        self.tracks_cam2 = {}
        self.track_pairs = defaultdict(lambda: defaultdict(int))  # Maps cam1_id to cam2_id with confidence count

        # Initialize scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda")

    def _initialize_detector(self) -> torch.nn.Module:
        detector = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
        detector.agnostic = True
        detector.classes = [0]  # Assuming 'person' class
        detector.conf = self.config.conf_threshold
        detector.to(self.config.device)
        detector.eval()  # Set to evaluation mode
        for param in detector.parameters():
            param.requires_grad = False
        return detector

    def _initialize_reid(self) -> torchreid.utils.FeatureExtractor:
        extractor = torchreid.utils.FeatureExtractor(
            model_name=self.config.reid_model_name,
            model_path=self.config.reid_weights_path,
            device=self.config.device
        )
        extractor.model.eval()
        return extractor

    def _calculate_feature_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two features"""
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

    def _update_tracks(self, tracks_dict, detections, features, camera_id):
        """Update tracks for a single camera"""
        if len(detections) == 0:
            return []

        # Calculate similarities between existing tracks and new detections
        track_indices = list(tracks_dict.keys())
        detection_indices = list(range(len(detections)))
        
        if len(track_indices) > 0 and len(features) > 0:
            similarity_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)
            
            for i, track_id in enumerate(track_indices):
                track = tracks_dict[track_id]
                track_feature = track.get_average_feature()
                
                # Efficient similarity computation using vectorization
                similarities = np.dot(features, track_feature) / (np.linalg.norm(features, axis=1) * np.linalg.norm(track_feature))
                similarity_matrix[i, :] = similarities

            # Hungarian algorithm matching
            matched_indices = []
            if similarity_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
                for r, c in zip(row_ind, col_ind):
                    if similarity_matrix[r, c] >= self.config.cos_threshold:
                        matched_indices.append((track_indices[r], c))

            # Update matched tracks
            unmatched_detections = set(detection_indices)
            for track_id, det_idx in matched_indices:
                tracks_dict[track_id].update(features[det_idx], detections[det_idx])
                unmatched_detections.discard(det_idx)

            # Handle unmatched detections
            for det_idx in unmatched_detections:
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                self.next_track_id += 1

        else:
            # Initialize new tracks for all detections
            for det_idx in range(len(detections)):
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                self.next_track_id += 1

        # Remove old tracks
        track_ids = list(tracks_dict.keys())
        for track_id in track_ids:
            tracks_dict[track_id].missed_frames += 1
            if tracks_dict[track_id].missed_frames > self.config.max_age:
                del tracks_dict[track_id]

        return list(tracks_dict.keys())

    def _match_across_cameras(self, features1, features2):
        """Match detections across cameras using ReID features"""
        if len(features1) == 0 or len(features2) == 0:
            return []

        similarity_matrix = features1 @ features2.T
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if similarity_matrix[r, c] >= self.config.cos_threshold:
                matches.append((r, c))
                
        return matches

    def extract_features_batch(self, frames: List[np.ndarray], detections_list: List[np.ndarray]) -> List[np.ndarray]:
        all_features = []
        
        # Batch processing of crops
        crops = []
        valid_indices = []
        for idx, (frame, detections) in enumerate(zip(frames, detections_list)):
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (128, 256))  # ReID model input size
                crops.append(crop_resized)
                valid_indices.append(idx)

        if not crops:
            return [np.array([]) for _ in frames]

        # Convert to tensor and batch
        crops_tensor = torch.from_numpy(np.array(crops)).permute(0, 3, 1, 2).float() / 255.0  # Normalize
        crops_tensor = crops_tensor.to(self.config.device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            try:
                features = self.reid_extractor(crops_tensor)
                features = features.cpu().numpy()
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
            except Exception as e:
                logger.warning(f"Failed to extract features: {e}")
                features = np.array([])

        # Assign features back to corresponding frames
        feature_dict = defaultdict(list)
        for idx, feat in zip(valid_indices, features):
            feature_dict[idx].append(feat)
        
        for i in range(len(frames)):
            if i in feature_dict:
                all_features.append(np.array(feature_dict[i]))
            else:
                all_features.append(np.array([]))
        
        return all_features

    def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
        # Initialize frame queues and readers
        queue1 = Queue(maxsize=10)
        queue2 = Queue(maxsize=10)
        reader1 = FrameReader(video1_path, queue1, name='Camera1')
        reader2 = FrameReader(video2_path, queue2, name='Camera2')
        reader1.start()
        reader2.start()

        video_writer = None
        frames_processed = 0

        try:
            while True:
                frame1 = queue1.get()
                frame2 = queue2.get()

                if frame1 is None or frame2 is None:
                    break

                start_time = time.time()
                
                frames_rgb = [
                    cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                ]

                # Batch detection with mixed precision
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    detections = self.detector(frames_rgb)
                
                dets1 = detections.xyxy[0].cpu().numpy()
                dets2 = detections.xyxy[1].cpu().numpy()

                detections_list = [dets1, dets2]

                # Extract features
                features_list = self.extract_features_batch(frames_rgb, detections_list)
                
                # Update tracks for each camera
                active_tracks1 = self._update_tracks(self.tracks_cam1, dets1, features_list[0], 1)
                active_tracks2 = self._update_tracks(self.tracks_cam2, dets2, features_list[1], 2)

                # Match tracks across cameras
                cross_camera_matches = []
                if active_tracks1 and active_tracks2:
                    features1 = np.array([self.tracks_cam1[tid].get_average_feature() for tid in active_tracks1])
                    features2 = np.array([self.tracks_cam2[tid].get_average_feature() for tid in active_tracks2])
                    matches = self._match_across_cameras(features1, features2)
                    
                    for idx1, idx2 in matches:
                        track1_id = active_tracks1[idx1]
                        track2_id = active_tracks2[idx2]
                        self.track_pairs[track1_id][track2_id] += 1
                        cross_camera_matches.append((track1_id, track2_id))

                # Visualize results
                vis = self.visualize(frame1, frame2, self.tracks_cam1, self.tracks_cam2, cross_camera_matches)

                if output_path and video_writer is None:
                    h, w = vis.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, (w, h), True
                    )

                if video_writer is not None:
                    video_writer.write(vis)

                # Calculate and display FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.namedWindow("Multi-Camera Tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Multi-Camera Tracking", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frames_processed += 1

        finally:
            reader1.stop()
            reader2.stop()
            reader1.join()
            reader2.join()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

    def visualize(self, frame1: np.ndarray, frame2: np.ndarray, 
                 tracks1: Dict, tracks2: Dict, 
                 matches: List[Tuple[int, int]]) -> np.ndarray:
        """Visualize tracks with consistent IDs"""
        vis1, vis2 = frame1.copy(), frame2.copy()
        
        # Create a mapping of track pairs for visualization
        matched_ids = {}
        for t1_id, t2_id in matches:
            if self.track_pairs[t1_id][t2_id] >= 3:  # Threshold for stable matching
                matched_ids[t1_id] = t2_id

        # Draw tracks for camera 1
        for track_id, track in tracks1.items():
            if track.consecutive_hits >= self.config.min_hits:
                color = tuple(map(int, self.colors[track_id % len(self.colors)]))
                box = track.last_position.astype(np.int32)
                cv2.rectangle(vis1, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis1, str(track_id), (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw tracks for camera 2
        for track_id, track in tracks2.items():
            if track.consecutive_hits >= self.config.min_hits:
                # Use same color for matched tracks
                matching_id = track_id
                for t1_id, t2_id in matched_ids.items():
                    if t2_id == track_id:
                        matching_id = t1_id
                        break
                color = tuple(map(int, self.colors[matching_id % len(self.colors)]))
                box = track.last_position.astype(np.int32)
                cv2.rectangle(vis2, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis2, str(matching_id), (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Concatenate both frames horizontally
        return np.hstack([vis1, vis2])

def main():
    import argparse
    
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


# 13 to 15FPS