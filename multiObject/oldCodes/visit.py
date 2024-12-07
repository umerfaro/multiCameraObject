import torch
import numpy as np
import cv2
import torchreid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from scipy.optimize import linear_sum_assignment
import time
from torch import amp  # Updated import
from collections import defaultdict
import threading
from queue import Queue
from ultralytics import YOLO  # Updated to use YOLOv8

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
    input_size: Tuple[int, int] = (416,416)
    batch_size: int = 128  # Increased batch size for better GPU utilization
    max_feature_distance: float = 0.7
    reid_keep_frames: int = 30  # Number of frames to keep ReID features

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
        if len(self.features) > 30:  # Keep only recent features
            self.features.pop(0)
        self.last_detection = detection
        self.missed_frames = 0  # Reset missed frames on successful update
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
        self.colors = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")  # Increased color palette
        self.last_time = time.time()
        
        # Track management
        self.next_track_id = 0
        self.tracks_cam1 = {}
        self.tracks_cam2 = {}
        self.track_pairs = defaultdict(lambda: defaultdict(int))  # Maps cam1_id to cam2_id with confidence count

        # **Global Tracking Enhancements**
        self.global_id = 0  # Global unique ID counter
        self.track_id_map_cam1 = {}  # Maps camera1 track IDs to global IDs
        self.track_id_map_cam2 = {}  # Maps camera2 track IDs to global IDs
        self.total_visit_count = 0  # Total unique visit count

        # Preallocate CUDA streams for overlapping operations
        self.det_stream = torch.cuda.Stream()
        self.reid_stream = torch.cuda.Stream()

        # Initialize visualization queue and thread
        self.vis_queue = Queue(maxsize=10)
        self.vis_thread = threading.Thread(target=self._visualization_worker, daemon=True)
        self.vis_thread.start()

    def _initialize_detector(self) -> YOLO:
        # Initialize YOLOv8 model with FP16
        detector = YOLO('yolov8n.pt')  # Use a lightweight YOLOv8 model
        detector.to(self.config.device)
        detector.eval()
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
            # Increment missed frames for all tracks
            to_delete = []
            for track_id, track in tracks_dict.items():
                track.missed_frames += 1
                if track.missed_frames > self.config.max_age:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del tracks_dict[track_id]
            return list(tracks_dict.keys())

        if len(tracks_dict) > 0 and len(features) > 0:
            # Create feature matrix for existing tracks and detections
            track_ids = list(tracks_dict.keys())
            track_features = np.array([tracks_dict[tid].get_average_feature() for tid in track_ids])
            detection_features = features

            # Normalize features
            track_features_norm = track_features / np.linalg.norm(track_features, axis=1, keepdims=True)
            detection_features_norm = detection_features / np.linalg.norm(detection_features, axis=1, keepdims=True)

            # Compute cosine similarity matrix
            similarity_matrix = np.dot(track_features_norm, detection_features_norm.T)

            # Perform matching using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            matches = []
            for r, c in zip(row_ind, col_ind):
                if similarity_matrix[r, c] >= self.config.cos_threshold:
                    matches.append((track_ids[r], c))

            # Update matched tracks
            matched_track_ids = set()
            matched_detection_ids = set()
            for track_id, det_idx in matches:
                tracks_dict[track_id].update(features[det_idx], detections[det_idx])
                matched_track_ids.add(track_id)
                matched_detection_ids.add(det_idx)

            # Create new tracks for unmatched detections
            unmatched_detections = set(range(len(detections))) - matched_detection_ids
            for det_idx in unmatched_detections:
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                self.next_track_id += 1

                # **Assign Global Unique ID for New Track**
                if camera_id == 1:
                    self.track_id_map_cam1[self.next_track_id - 1] = self.global_id
                else:
                    self.track_id_map_cam2[self.next_track_id - 1] = self.global_id
                self.global_id += 1
                self.total_visit_count += 1  # Increment total visit count

            # Increment missed frames for unmatched tracks
            unmatched_tracks = set(tracks_dict.keys()) - matched_track_ids
            to_delete = []
            for track_id in unmatched_tracks:
                tracks_dict[track_id].missed_frames += 1
                if tracks_dict[track_id].missed_frames > self.config.max_age:
                    del tracks_dict[track_id]
            # Note: No need to return here, as tracks are already updated

        else:
            # Initialize new tracks for all detections
            for det_idx in range(len(detections)):
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track

                # **Assign Global Unique ID for New Track**
                if camera_id == 1:
                    self.track_id_map_cam1[self.next_track_id] = self.global_id
                else:
                    self.track_id_map_cam2[self.next_track_id] = self.global_id
                self.global_id += 1
                self.total_visit_count += 1  # Increment total visit count

                self.next_track_id += 1

        return list(tracks_dict.keys())

    def _match_across_cameras(self, features1, features2):
        """Match tracks across cameras using ReID features"""
        if len(features1) == 0 or len(features2) == 0:
            return []

        # Normalize features
        features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(features1_norm, features2_norm.T)

        # Perform matching using Hungarian algorithm
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
        frame_indices = []
        det_indices = []
        for idx, (frame, detections) in enumerate(zip(frames, detections_list)):
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det[:4])
                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (128, 256))  # ReID model input size
                crops.append(crop_resized)
                frame_indices.append(idx)
                det_indices.append(det_idx)

        if not crops:
            return [np.array([]) for _ in frames]

        # Convert to tensor and batch
        crops_tensor = torch.from_numpy(np.array(crops)).permute(0, 3, 1, 2).float() / 255.0  # Normalize
        crops_tensor = crops_tensor.to(self.config.device).half()  # Convert to FP16

        features = []
        with torch.no_grad():
            with amp.autocast(device_type=self.config.device):  # Updated usage
                try:
                    # Use separate CUDA streams for ReID
                    with torch.cuda.stream(self.reid_stream):
                        features_tensor = self.reid_extractor.model(crops_tensor)
                        features_tensor = features_tensor.cpu()
                    torch.cuda.current_stream().wait_stream(self.reid_stream)
                    features = features_tensor.numpy()
                    features = features / np.linalg.norm(features, axis=1, keepdims=True)
                except Exception as e:
                    logger.warning(f"Failed to extract features: {e}")
                    features = np.array([])

        # Assign features back to corresponding frames
        feature_dict = defaultdict(list)
        for frame_idx, feat in zip(frame_indices, features):
            feature_dict[frame_idx].append(feat)

        for i in range(len(frames)):
            if i in feature_dict:
                all_features.append(np.array(feature_dict[i]))
            else:
                all_features.append(np.array([]))

        return all_features

    def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
        # Initialize frame queues and readers
        queue1 = Queue(maxsize=20)  # Increased queue size to buffer more frames
        queue2 = Queue(maxsize=20)
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

                # Batch detection for both frames together
                with torch.cuda.stream(self.det_stream):
                    with amp.autocast(device_type=self.config.device):  # Updated usage
                        # **Specify classes=[0] to detect only 'person'**
                        detections = self.detector(frames_rgb, device=self.config.device, half=True, classes=[0])

                torch.cuda.current_stream().wait_stream(self.det_stream)

                # Parse detections
                dets1 = detections[0].boxes.xyxy.cpu().numpy() if len(detections) > 0 else np.array([])
                dets2 = detections[1].boxes.xyxy.cpu().numpy() if len(detections) > 1 else np.array([])

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

                        # **Assign Global Unique IDs Based on Cross-Camera Matches**
                        global_id_cam1 = self.track_id_map_cam1.get(track1_id)
                        global_id_cam2 = self.track_id_map_cam2.get(track2_id)

                        if global_id_cam1 is None and global_id_cam2 is None:
                            # Both tracks are new, assign a new global ID
                            self.track_id_map_cam1[track1_id] = self.global_id
                            self.track_id_map_cam2[track2_id] = self.global_id
                            self.global_id += 1
                            self.total_visit_count += 1
                        elif global_id_cam1 is not None and global_id_cam2 is None:
                            # Assign camera2 track to camera1's global ID
                            self.track_id_map_cam2[track2_id] = global_id_cam1
                        elif global_id_cam1 is None and global_id_cam2 is not None:
                            # Assign camera1 track to camera2's global ID
                            self.track_id_map_cam1[track1_id] = global_id_cam2
                        else:
                            # Both have global IDs, ensure they are the same
                            if global_id_cam1 != global_id_cam2:
                                # Handle conflicting global IDs if necessary
                                # For simplicity, we'll keep camera1's global ID and update camera2's
                                self.track_id_map_cam2[track2_id] = global_id_cam1

                # Prepare visualization data
                vis_data = (frame1, frame2, self.tracks_cam1, self.tracks_cam2, cross_camera_matches)
                self.vis_queue.put(vis_data)  # Offload visualization to a separate thread

                if output_path:
                    if video_writer is None:
                        h, w = frame1.shape[:2]
                        video_writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, (w * 2, h), True
                        )
                    # The visualization thread handles writing to the video_writer

                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                frames_processed += 1

        except Exception as e:
            logger.error(f"An error occurred during video processing: {e}")
        finally:
            reader1.stop()
            reader2.stop()
            reader1.join()
            reader2.join()
            self.vis_queue.put(None)  # Signal visualization thread to terminate
            self.vis_thread.join()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Total frames processed: {frames_processed}")
            logger.info(f"Total unique visits: {self.total_visit_count}")  # Log total visit count

    def _visualization_worker(self):
        """Separate thread for handling visualization to avoid blocking main processing."""
        video_writer = None
        while True:
            vis_data = self.vis_queue.get()
            if vis_data is None:
                break
            frame1, frame2, tracks1, tracks2, matches = vis_data
            vis = self.visualize(frame1, frame2, tracks1, tracks2, matches)

            if video_writer is None and hasattr(self, 'config') and self.config:
                # Initialize video writer if output path is provided
                pass  # Handled in process_videos

            # Calculate and display FPS on the visualization
            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - self.last_time)
            self.last_time = current_time
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # **Display Total Visit Count**
            cv2.putText(vis, f"Total Visits: {self.total_visit_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Resize visualization to reduce rendering time
            display_vis = cv2.resize(vis, (1280, 720))  # Adjust as needed

            cv2.namedWindow("Multi-Camera Tracking", cv2.WINDOW_NORMAL)
            cv2.imshow("Multi-Camera Tracking", display_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
                # Get global ID
                global_id = self.track_id_map_cam1.get(track_id, track_id)
                color = tuple(map(int, self.colors[global_id % len(self.colors)]))
                box = track.last_position.astype(int)
                cv2.rectangle(vis1, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis1, f"ID: {global_id}", (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw tracks for camera 2
        for track_id, track in tracks2.items():
            if track.consecutive_hits >= self.config.min_hits:
                # Get global ID
                global_id = self.track_id_map_cam2.get(track_id, track_id)
                color = tuple(map(int, self.colors[global_id % len(self.colors)]))
                box = track.last_position.astype(int)
                cv2.rectangle(vis2, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis2, f"ID: {global_id}", (box[0], box[1]-10), 
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