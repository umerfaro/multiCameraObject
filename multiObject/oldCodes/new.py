import torch
import numpy as np
import cv2
import torchreid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from scipy.optimize import linear_sum_assignment
import time
from collections import defaultdict
import threading
from queue import Queue
from ultralytics import YOLO
from torch.amp import autocast  # Updated import for autocast usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    conf_threshold: float = 0.30
    cos_threshold: float = 0.80
    reid_model_name: str = "osnet_x0_25"
    reid_weights_path: str = "./weights/osnet_x0_25.pth.tar"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (320, 320)  # Reduced resolution for faster inference
    batch_size: int = 64
    max_feature_distance: float = 0.7
    reid_keep_frames: int = 30

class Track:
    def __init__(self, feature, detection, track_id):
        self.id = track_id
        self.features = [feature] if feature.size > 0 else []
        self.last_detection = detection
        self.missed_frames = 0
        self.consecutive_hits = 1
        self.last_position = detection[:4]

    def update(self, feature, detection):
        if feature.size > 0:
            self.features.append(feature)
            if len(self.features) > 30:
                self.features.pop(0)
        self.last_detection = detection
        self.missed_frames = 0
        self.consecutive_hits += 1
        self.last_position = detection[:4]

    def get_average_feature(self):
        if not self.features:
            # Assuming 512-dim feature (adjust if your model differs)
            return np.zeros((512,), dtype=np.float32)
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
        self.colors = np.random.randint(0, 255, size=(10000, 3), dtype="uint8")
        self.last_time = time.time()

        self.next_track_id = 0
        self.tracks_cam1 = {}
        self.tracks_cam2 = {}
        self.track_pairs = defaultdict(lambda: defaultdict(int))

        self.global_id = 0
        self.track_id_map_cam1 = {}
        self.track_id_map_cam2 = {}
        self.total_visit_count = 0

        self.det_stream = torch.cuda.Stream()
        self.reid_stream = torch.cuda.Stream()

        self.vis_queue = Queue(maxsize=10)
        self.vis_thread = threading.Thread(target=self._visualization_worker, daemon=True)
        self.vis_thread.start()

        self.frame_counter = 0

    def _initialize_detector(self) -> YOLO:
        detector = YOLO('yolov8n.pt')
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

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features and avoid division by zero issues."""
        if features.size == 0:
            return features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        return features / norms

    def _update_tracks(self, tracks_dict, detections, features, camera_id):
        if len(detections) == 0:
            # Increment missed frames for all existing tracks
            to_delete = []
            for track_id, track in tracks_dict.items():
                track.missed_frames += 1
                if track.missed_frames > self.config.max_age:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del tracks_dict[track_id]
            return list(tracks_dict.keys())

        if len(tracks_dict) > 0 and features.size > 0:
            track_ids = list(tracks_dict.keys())
            track_features = np.array([tracks_dict[tid].get_average_feature() for tid in track_ids])
            track_features_norm = self._normalize_features(track_features)
            detection_features_norm = self._normalize_features(features)

            similarity_matrix = np.dot(track_features_norm, detection_features_norm.T)
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

            matches = [(track_ids[r], c) for r, c in zip(row_ind, col_ind) 
                       if similarity_matrix[r, c] >= self.config.cos_threshold]

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

                if camera_id == 1:
                    self.track_id_map_cam1[self.next_track_id] = self.global_id
                else:
                    self.track_id_map_cam2[self.next_track_id] = self.global_id
                self.global_id += 1
                self.total_visit_count += 1

                self.next_track_id += 1

            # Increment missed frames for unmatched tracks
            unmatched_tracks = set(tracks_dict.keys()) - matched_track_ids
            to_delete = []
            for track_id in unmatched_tracks:
                tracks_dict[track_id].missed_frames += 1
                if tracks_dict[track_id].missed_frames > self.config.max_age:
                    to_delete.append(track_id)
            for tid in to_delete:
                del tracks_dict[tid]

        else:
            # Initialize new tracks if no existing tracks or no features
            for det_idx in range(len(detections)):
                feat = features[det_idx] if features.size > 0 else np.array([])
                new_track = Track(feat, detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                if camera_id == 1:
                    self.track_id_map_cam1[self.next_track_id] = self.global_id
                else:
                    self.track_id_map_cam2[self.next_track_id] = self.global_id
                self.global_id += 1
                self.total_visit_count += 1
                self.next_track_id += 1

        return list(tracks_dict.keys())

    def _match_across_cameras(self, features1, features2):
        if features1.size == 0 or features2.size == 0:
            return []
        f1_norm = self._normalize_features(features1)
        f2_norm = self._normalize_features(features2)

        sim_matrix = np.dot(f1_norm, f2_norm.T)
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if sim_matrix[r, c] >= self.config.cos_threshold]
        return matches

    def extract_features_batch(self, frames: List[np.ndarray], detections_list: List[np.ndarray]) -> List[np.ndarray]:
        crops = []
        frame_indices = []
        det_indices = []
        for idx, (frame, detections) in enumerate(zip(frames, detections_list)):
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2 = map(int, det[:4])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (128, 256))
                crops.append(crop_resized)
                frame_indices.append(idx)
                det_indices.append(det_idx)

        if not crops:
            return [np.array([]) for _ in frames]

        crops_tensor = torch.from_numpy(np.array(crops)).permute(0, 3, 1, 2).float() / 255.0
        crops_tensor = crops_tensor.to(self.config.device).half()

        with torch.no_grad():
            try:
                with torch.cuda.stream(self.reid_stream):
                    with autocast("cuda"):
                        features_tensor = self.reid_extractor.model(crops_tensor)
                    features_tensor = features_tensor.cpu()
                torch.cuda.current_stream().wait_stream(self.reid_stream)
                features = features_tensor.numpy()
                # Already normalized in the model, but let's ensure norms are not zero
                # (Though reid_extractor.model outputs normalized features typically)
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                norms[norms == 0] = 1e-6
                features = features / norms
            except Exception as e:
                logger.warning(f"Failed to extract features: {e}")
                return [np.array([]) for _ in frames]

        feature_dict = defaultdict(list)
        for f_idx, feat in zip(frame_indices, features):
            feature_dict[f_idx].append(feat)

        all_features = []
        for i in range(len(frames)):
            if i in feature_dict:
                all_features.append(np.array(feature_dict[i]))
            else:
                all_features.append(np.array([]))
        return all_features

    def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
        queue1 = Queue(maxsize=20)
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

                self.frame_counter += 1
                start_time = time.time()

                f1_resized = cv2.resize(frame1, self.config.input_size)
                f2_resized = cv2.resize(frame2, self.config.input_size)
                frames_rgb = [
                    cv2.cvtColor(f1_resized, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(f2_resized, cv2.COLOR_BGR2RGB)
                ]

                with torch.cuda.stream(self.det_stream):
                    with autocast("cuda"):
                        detections = self.detector(frames_rgb, device=self.config.device, half=True, classes=[0])
                torch.cuda.current_stream().wait_stream(self.det_stream)

                dets1 = detections[0].boxes.xyxy.cpu().numpy() if len(detections) > 0 else np.array([])
                dets2 = detections[1].boxes.xyxy.cpu().numpy() if len(detections) > 1 else np.array([])

                # Extract features every 2 frames
                if self.frame_counter % 2 == 0:
                    features_list = self.extract_features_batch(frames_rgb, [dets1, dets2])
                else:
                    features_list = [np.array([]), np.array([])]

                active_tracks1 = self._update_tracks(self.tracks_cam1, dets1, features_list[0], 1)
                active_tracks2 = self._update_tracks(self.tracks_cam2, dets2, features_list[1], 2)

                # Cross-camera matching every 5 frames
                cross_camera_matches = []
                if self.frame_counter % 5 == 0 and active_tracks1 and active_tracks2:
                    features1 = np.array([self.tracks_cam1[tid].get_average_feature() for tid in active_tracks1])
                    features2 = np.array([self.tracks_cam2[tid].get_average_feature() for tid in active_tracks2])
                    matches = self._match_across_cameras(features1, features2)
                    for idx1, idx2 in matches:
                        track1_id = active_tracks1[idx1]
                        track2_id = active_tracks2[idx2]
                        self.track_pairs[track1_id][track2_id] += 1
                        cross_camera_matches.append((track1_id, track2_id))

                        global_id_cam1 = self.track_id_map_cam1.get(track1_id)
                        global_id_cam2 = self.track_id_map_cam2.get(track2_id)

                        if global_id_cam1 is None and global_id_cam2 is None:
                            self.track_id_map_cam1[track1_id] = self.global_id
                            self.track_id_map_cam2[track2_id] = self.global_id
                            self.global_id += 1
                            self.total_visit_count += 1
                        elif global_id_cam1 is not None and global_id_cam2 is None:
                            self.track_id_map_cam2[track2_id] = global_id_cam1
                        elif global_id_cam1 is None and global_id_cam2 is not None:
                            self.track_id_map_cam1[track1_id] = global_id_cam2
                        else:
                            if global_id_cam1 != global_id_cam2:
                                self.track_id_map_cam2[track2_id] = global_id_cam1

                # Visualization every 5 frames
                if self.frame_counter % 5 == 0 and not self.vis_queue.full():
                    vis_data = (frame1, frame2, self.tracks_cam1, self.tracks_cam2, cross_camera_matches)
                    self.vis_queue.put(vis_data)

                fps = 1.0 / (time.time() - start_time)
                frames_processed += 1

        except Exception as e:
            logger.error(f"An error occurred during video processing: {e}")
        finally:
            reader1.stop()
            reader2.stop()
            reader1.join()
            reader2.join()
            self.vis_queue.put(None)
            self.vis_thread.join()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Total frames processed: {frames_processed}")
            logger.info(f"Total unique visits: {self.total_visit_count}")

    def _visualization_worker(self):
        while True:
            vis_data = self.vis_queue.get()
            if vis_data is None:
                break
            frame1, frame2, tracks1, tracks2, matches = vis_data
            vis = self.visualize(frame1, frame2, tracks1, tracks2, matches)
            cv2.imshow("Multi-Camera Tracking", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def visualize(self, frame1: np.ndarray, frame2: np.ndarray, 
                  tracks1: Dict, tracks2: Dict, 
                  matches: List[Tuple[int, int]]) -> np.ndarray:
        vis1 = frame1.copy()
        vis2 = frame2.copy()

        matched_ids = {}
        for t1_id, t2_id in matches:
            if self.track_pairs[t1_id][t2_id] >= 3:
                matched_ids[t1_id] = t2_id

        for track_id, track in tracks1.items():
            if track.consecutive_hits >= self.config.min_hits:
                global_id = self.track_id_map_cam1.get(track_id, track_id)
                color = tuple(map(int, self.colors[global_id % len(self.colors)]))
                box = track.last_position.astype(int)
                cv2.rectangle(vis1, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis1, f"ID: {global_id}", (box[0], max(box[1]-10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        for track_id, track in tracks2.items():
            if track.consecutive_hits >= self.config.min_hits:
                global_id = self.track_id_map_cam2.get(track_id, track_id)
                color = tuple(map(int, self.colors[global_id % len(self.colors)]))
                box = track.last_position.astype(int)
                cv2.rectangle(vis2, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis2, f"ID: {global_id}", (box[0], max(box[1]-10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        vis = np.hstack([vis1, vis2])
        cv2.putText(vis, f"Total Visits: {self.total_visit_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis

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
