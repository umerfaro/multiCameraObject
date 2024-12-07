# import torch
# import numpy as np
# import cv2
# import torchreid
# from dataclasses import dataclass
# from typing import List, Tuple, Optional, Dict
# import logging
# from scipy.optimize import linear_sum_assignment
# import time
# from torch.cuda.amp import autocast
# from collections import defaultdict

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @dataclass
# class TrackingConfig:
#     iou_threshold: float = 0.3
#     max_age: int = 30
#     min_hits: int = 3
#     conf_threshold: float = 0.30
#     cos_threshold: float = 0.80
#     reid_model_name: str = "osnet_x1_0"
#     reid_weights_path: str = "./weights/osnet_x1_0.pth.tar"
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     input_size: Tuple[int, int] = (640, 640)
#     batch_size: int = 2
#     max_feature_distance: float = 0.7
#     reid_keep_frames: int = 30  # Number of frames to keep ReID features

# class Track:
#     def __init__(self, feature, detection, track_id):
#         self.id = track_id
#         self.features = [feature]
#         self.last_detection = detection
#         self.missed_frames = 0
#         self.consecutive_hits = 1
#         self.last_position = detection[:4]

#     def update(self, feature, detection):
#         self.features.append(feature)
#         if len(self.features) > 30:  # Keep only recent features
#             self.features.pop(0)
#         self.last_detection = detection
#         self.missed_frames = 0
#         self.consecutive_hits += 1
#         self.last_position = detection[:4]

#     def get_average_feature(self):
#         return np.mean(self.features, axis=0)

# class MultiCameraTracker:
#     def __init__(self, config: TrackingConfig):
#         self.config = config
#         self.detector = self._initialize_detector()
#         self.reid_extractor = self._initialize_reid()
#         self.colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")
#         self.last_time = time.time()
        
#         # Track management
#         self.next_track_id = 0
#         self.tracks_cam1 = {}
#         self.tracks_cam2 = {}
#         self.track_pairs = defaultdict(lambda: defaultdict(int))  # Maps cam1_id to cam2_id with confidence count
        
#     def _initialize_detector(self) -> torch.nn.Module:
#         detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
#         detector.agnostic = True
#         detector.classes = [0]
#         detector.conf = self.config.conf_threshold
#         detector.to(self.config.device)
#         for param in detector.parameters():
#             param.requires_grad = False
#         return detector

#     def _initialize_reid(self) -> torchreid.utils.FeatureExtractor:
#         extractor = torchreid.utils.FeatureExtractor(
#             model_name=self.config.reid_model_name,
#             model_path=self.config.reid_weights_path,
#             device=self.config.device
#         )
#         extractor.model.eval()
#         return extractor

#     def _calculate_feature_similarity(self, feature1, feature2):
#         """Calculate cosine similarity between two features"""
#         return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

#     def _update_tracks(self, tracks_dict, detections, features, camera_id):
#         """Update tracks for a single camera"""
#         if len(detections) == 0:
#             return []

#         # Calculate similarities between existing tracks and new detections
#         track_indices = list(tracks_dict.keys())
#         detection_indices = list(range(len(detections)))
        
#         if len(track_indices) > 0:
#             similarity_matrix = np.zeros((len(track_indices), len(detection_indices)))
            
#             for i, track_id in enumerate(track_indices):
#                 track = tracks_dict[track_id]
#                 track_feature = track.get_average_feature()
                
#                 for j, feat in enumerate(features):
#                     similarity_matrix[i, j] = self._calculate_feature_similarity(track_feature, feat)

#             # Hungarian algorithm matching
#             matched_indices = []
#             if similarity_matrix.size > 0:
#                 row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
#                 for r, c in zip(row_ind, col_ind):
#                     if similarity_matrix[r, c] >= self.config.cos_threshold:
#                         matched_indices.append((track_indices[r], detection_indices[c]))

#             # Update matched tracks
#             unmatched_detections = set(detection_indices)
#             for track_id, det_idx in matched_indices:
#                 tracks_dict[track_id].update(features[det_idx], detections[det_idx])
#                 unmatched_detections.remove(det_idx)

#             # Handle unmatched detections
#             for det_idx in unmatched_detections:
#                 new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
#                 tracks_dict[self.next_track_id] = new_track
#                 self.next_track_id += 1

#         else:
#             # Initialize new tracks for all detections
#             for det_idx in range(len(detections)):
#                 new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
#                 tracks_dict[self.next_track_id] = new_track
#                 self.next_track_id += 1

#         # Remove old tracks
#         track_ids = list(tracks_dict.keys())
#         for track_id in track_ids:
#             tracks_dict[track_id].missed_frames += 1
#             if tracks_dict[track_id].missed_frames > self.config.max_age:
#                 del tracks_dict[track_id]

#         return list(tracks_dict.keys())

#     def _match_across_cameras(self, features1, features2):
#         """Match detections across cameras using ReID features"""
#         if len(features1) == 0 or len(features2) == 0:
#             return []

#         similarity_matrix = features1 @ features2.T
#         row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
#         matches = []
#         for r, c in zip(row_ind, col_ind):
#             if similarity_matrix[r, c] >= self.config.cos_threshold:
#                 matches.append((r, c))
                
#         return matches

#     def extract_features_batch(self, frames: List[np.ndarray], detections_list: List[np.ndarray]) -> List[np.ndarray]:
#         all_features = []
        
#         for frame, detections in zip(frames, detections_list):
#             if len(detections) == 0:
#                 all_features.append(np.array([]))
#                 continue

#             crops = []
#             for det in detections:
#                 x1, y1, x2, y2 = map(int, det[:4])
#                 crop = frame[max(0, y1):min(frame.shape[0], y2), 
#                            max(0, x1):min(frame.shape[1], x2)]
#                 if crop.size == 0:
#                     continue
#                 crops.append(cv2.resize(crop, (128, 256)))  # ReID model input size

#             if not crops:
#                 all_features.append(np.array([]))
#                 continue

#             with torch.no_grad(), autocast(enabled=True):
#                 try:
#                     features = self.reid_extractor(crops)
#                     features = features.cpu().numpy()
#                     features = features / np.linalg.norm(features, axis=1, keepdims=True)
#                     all_features.append(features)
#                 except Exception as e:
#                     logger.warning(f"Failed to extract features: {e}")
#                     all_features.append(np.array([]))

#         return all_features

#     def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
#         video1 = cv2.VideoCapture(video1_path)
#         video2 = cv2.VideoCapture(video2_path)
#         video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

#         num_frames = min(int(video1.get(cv2.CAP_PROP_FRAME_COUNT)),
#                         int(video2.get(cv2.CAP_PROP_FRAME_COUNT)))
        
#         video_writer = None
#         frames_processed = 0

#         try:
#             while frames_processed < num_frames:
#                 start_time = time.time()
                
#                 ret1, frame1 = video1.read()
#                 ret2, frame2 = video2.read()
#                 if not ret1 or not ret2:
#                     break

#                 frames_rgb = [
#                     cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
#                     cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
#                 ]

#                 with torch.no_grad():
#                     detections = self.detector(frames_rgb)

#                 dets1 = detections.xyxy[0].cpu().numpy()
#                 dets2 = detections.xyxy[1].cpu().numpy()

#                 features_list = self.extract_features_batch(frames_rgb, [dets1, dets2])
                
#                 # Update tracks for each camera
#                 active_tracks1 = self._update_tracks(self.tracks_cam1, dets1, features_list[0], 1)
#                 active_tracks2 = self._update_tracks(self.tracks_cam2, dets2, features_list[1], 2)

#                 # Match tracks across cameras
#                 cross_camera_matches = []
#                 if active_tracks1 and active_tracks2:
#                     features1 = np.array([self.tracks_cam1[tid].get_average_feature() for tid in active_tracks1])
#                     features2 = np.array([self.tracks_cam2[tid].get_average_feature() for tid in active_tracks2])
#                     matches = self._match_across_cameras(features1, features2)
                    
#                     for idx1, idx2 in matches:
#                         track1_id = active_tracks1[idx1]
#                         track2_id = active_tracks2[idx2]
#                         self.track_pairs[track1_id][track2_id] += 1
#                         cross_camera_matches.append((track1_id, track2_id))

#                 # Visualize results
#                 vis = self.visualize(frame1, frame2, self.tracks_cam1, self.tracks_cam2, cross_camera_matches)

#                 if output_path and video_writer is None:
#                     h, w = vis.shape[:2]
#                     video_writer = cv2.VideoWriter(
#                         output_path,
#                         cv2.VideoWriter_fourcc(*'MJPG'),
#                         30, (w, h), True
#                     )

#                 if video_writer is not None:
#                     video_writer.write(vis)

#                 # Calculate and display FPS
#                 fps = 1.0 / (time.time() - start_time)
#                 cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 cv2.namedWindow("Multi-Camera Tracking", cv2.WINDOW_NORMAL)
#                 cv2.imshow("Multi-Camera Tracking", vis)
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # frames_processed += 1

#         finally:
#             video1.release()
#             video2.release()
#             if video_writer is not None:
#                 video_writer.release()
#             cv2.destroyAllWindows()

#     def visualize(self, frame1: np.ndarray, frame2: np.ndarray, 
#                  tracks1: Dict, tracks2: Dict, 
#                  matches: List[Tuple[int, int]]) -> np.ndarray:
#         """Visualize tracks with consistent IDs"""
#         vis1, vis2 = frame1.copy(), frame2.copy()
        
#         # Create a mapping of track pairs for visualization
#         matched_ids = {}
#         for t1_id, t2_id in matches:
#             if self.track_pairs[t1_id][t2_id] >= 3:  # Threshold for stable matching
#                 matched_ids[t1_id] = t2_id

#         # Draw tracks for camera 1
#         for track_id, track in tracks1.items():
#             if track.consecutive_hits >= self.config.min_hits:
#                 color = tuple(map(int, self.colors[track_id % len(self.colors)]))
#                 box = track.last_position.astype(np.int32)
#                 cv2.rectangle(vis1, (box[0], box[1]), (box[2], box[3]), color, 2)
#                 cv2.putText(vis1, str(track_id), (box[0], box[1]-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         # Draw tracks for camera 2
#         for track_id, track in tracks2.items():
#             if track.consecutive_hits >= self.config.min_hits:
#                 # Use same color for matched tracks
#                 matching_id = track_id
#                 for t1_id, t2_id in matched_ids.items():
#                     if t2_id == track_id:
#                         matching_id = t1_id
#                         break
                        
#                 color = tuple(map(int, self.colors[matching_id % len(self.colors)]))
#                 box = track.last_position.astype(np.int32)
#                 cv2.rectangle(vis2, (box[0], box[1]), (box[2], box[3]), color, 2)
#                 cv2.putText(vis2, str(matching_id), (box[0], box[1]-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
#         return np.hstack([vis1, vis2])

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Multi-Camera Tracking System")
#     parser.add_argument("--video1", type=str, required=True)
#     parser.add_argument("--video2", type=str, required=True)
#     parser.add_argument("--output", type=str)
#     parser.add_argument("--conf", type=float, default=0.3)
#     parser.add_argument("--iou-thresh", type=float, default=0.3)
#     parser.add_argument("--cos-thresh", type=float, default=0.8)
    
#     args = parser.parse_args()
    
#     config = TrackingConfig(
#         conf_threshold=args.conf,
#         iou_threshold=args.iou_thresh,
#         cos_threshold=args.cos_thresh
#     )
#     tracker = MultiCameraTracker(config)
#     tracker.process_videos(args.video1, args.video2, args.output)   

# if __name__ == "__main__":
#     main()


# # python imrpove_code2.py --video1 cam11.mp4 --video2 cam44.mp4 --output output.avi
# # 8 to 10FPS



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
from torch2trt import torch2trt

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
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 2
    max_feature_distance: float = 0.7
    reid_keep_frames: int = 30

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
        if len(self.features) > 30:
            self.features.pop(0)
        self.last_detection = detection
        self.missed_frames = 0
        self.consecutive_hits += 1
        self.last_position = detection[:4]

    def get_average_feature(self):
        return np.mean(self.features, axis=0)

class FrameReader(threading.Thread):
    def __init__(self, video_path, queue, target_size=(640, 640), name=''):
        super().__init__()
        self.video = cv2.VideoCapture(video_path)
        self.queue = queue
        self.stopped = False
        self.name = name
        self.target_size = target_size

    def run(self):
        while not self.stopped:
            ret, frame = self.video.read()
            if not ret:
                self.stopped = True
                self.queue.put(None)
                break
            # Resize frame immediately after reading
            frame = cv2.resize(frame, self.target_size)
            self.queue.put(frame)

    def stop(self):
        self.stopped = True
        self.video.release()

class MultiCameraTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.detector = self._initialize_detector()
        self.reid_extractor = self._initialize_reid()
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")
        self.last_time = time.time()
        self.next_track_id = 0
        self.tracks_cam1 = {}
        self.tracks_cam2 = {}
        self.track_pairs = defaultdict(lambda: defaultdict(int))
        self.scaler = GradScaler()

    def _initialize_detector(self) -> torch.nn.Module:
        detector = torch.hub.load('ultralytics/yolov8', 'yolov8n')
        detector.conf = self.config.conf_threshold
        detector = detector.to(self.config.device).eval()
        
        # TensorRT optimization
        x = torch.randn(1, 3, 640, 640).to(self.config.device)
        detector_trt = torch2trt(detector, [x], 
                               fp16_mode=True,
                               max_workspace_size=1<<25)
        return detector_trt

    def _initialize_reid(self) -> torchreid.utils.FeatureExtractor:
        extractor = torchreid.utils.FeatureExtractor(
            model_name=self.config.reid_model_name,
            model_path=self.config.reid_weights_path,
            device=self.config.device
        )
        extractor.model.eval()
        return extractor

    def extract_features_batch(self, frames: List[np.ndarray], detections_list: List[np.ndarray]) -> List[np.ndarray]:
        max_detections = sum(len(d) for d in detections_list)
        if max_detections == 0:
            return [np.array([]) for _ in frames]

        crops_tensor = torch.zeros((max_detections, 3, 256, 128), 
                                 device=self.config.device)
        
        idx = 0
        valid_indices = []
        for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                crop = frame[max(0, y1):min(frame.shape[0], y2), 
                            max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue
                    
                crop_resized = cv2.resize(crop, (128, 256))
                crops_tensor[idx] = torch.from_numpy(crop_resized).permute(2,0,1)
                valid_indices.append(frame_idx)
                idx += 1

        if idx == 0:
            return [np.array([]) for _ in frames]

        crops_tensor = crops_tensor[:idx] / 255.0

        with torch.cuda.amp.autocast():
            features = self.reid_extractor(crops_tensor)
            features = features.cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

        feature_dict = defaultdict(list)
        for idx, feat in zip(valid_indices, features):
            feature_dict[idx].append(feat)
        
        return [np.array(feature_dict.get(i, [])) for i in range(len(frames))]

    def _update_tracks(self, tracks_dict, detections, features, camera_id):
        if len(detections) == 0:
            return []

        track_indices = list(tracks_dict.keys())
        detection_indices = list(range(len(detections)))
        
        if track_indices and features.size > 0:
            similarity_matrix = np.zeros((len(track_indices), len(detection_indices)))
            track_features = np.array([tracks_dict[tid].get_average_feature() for tid in track_indices])
            similarity_matrix = features @ track_features.T

            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            matched_indices = [
                (track_indices[r], c) for r, c in zip(row_ind, col_ind)
                if similarity_matrix[r, c] >= self.config.cos_threshold
            ]

            unmatched_detections = set(detection_indices)
            for track_id, det_idx in matched_indices:
                tracks_dict[track_id].update(features[det_idx], detections[det_idx])
                unmatched_detections.discard(det_idx)

            for det_idx in unmatched_detections:
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                self.next_track_id += 1

        else:
            for det_idx in range(len(detections)):
                new_track = Track(features[det_idx], detections[det_idx], self.next_track_id)
                tracks_dict[self.next_track_id] = new_track
                self.next_track_id += 1

        # Remove old tracks
        for track_id in list(tracks_dict.keys()):
            tracks_dict[track_id].missed_frames += 1
            if tracks_dict[track_id].missed_frames > self.config.max_age:
                del tracks_dict[track_id]

        return list(tracks_dict.keys())

    def _match_across_cameras(self, features1, features2):
        if len(features1) == 0 or len(features2) == 0:
            return []

        similarity_matrix = features1 @ features2.T
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        return [(r, c) for r, c in zip(row_ind, col_ind) 
                if similarity_matrix[r, c] >= self.config.cos_threshold]

    def process_videos(self, video1_path: str, video2_path: str, output_path: Optional[str] = None) -> None:
        queue1 = Queue(maxsize=10)
        queue2 = Queue(maxsize=10)
        reader1 = FrameReader(video1_path, queue1, self.config.input_size, 'Camera1')
        reader2 = FrameReader(video2_path, queue2, self.config.input_size, 'Camera2')
        reader1.start()
        reader2.start()

        video_writer = None
        frames_processed = 0
        skip_frames = 0  # Add frame skipping for performance

        try:
            while True:
                frame1 = queue1.get()
                frame2 = queue2.get()

                if frame1 is None or frame2 is None:
                    break

                if skip_frames > 0:
                    skip_frames -= 1
                    continue

                start_time = time.time()
                
                frames_rgb = [
                    cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                ]

                # Batch detection with mixed precision
                with torch.cuda.amp.autocast():
                    detections = self.detector(frames_rgb)
                
                dets1 = detections.xyxy[0].cpu().numpy()
                dets2 = detections.xyxy[1].cpu().numpy()

                # Extract features
                features_list = self.extract_features_batch(frames_rgb, [dets1, dets2])
                
                # Update tracks
                active_tracks1 = self._update_tracks(self.tracks_cam1, dets1, features_list[0], 1)
                active_tracks2 = self._update_tracks(self.tracks_cam2, dets2, features_list[1], 2)

                # Match across cameras
                cross_camera_matches = []
                if active_tracks1 and active_tracks2:
                    features1 = np.array([self.tracks_cam1[tid].get_average_feature() 
                                        for tid in active_tracks1])
                    features2 = np.array([self.tracks_cam2[tid].get_average_feature() 
                                        for tid in active_tracks2])
                    matches = self._match_across_cameras(features1, features2)
                    
                    for idx1, idx2 in matches:
                        track1_id = active_tracks1[idx1]
                        track2_id = active_tracks2[idx2]
                        self.track_pairs[track1_id][track2_id] += 1
                        cross_camera_matches.append((track1_id, track2_id))

                # Visualize
                vis = self.visualize(frame1, frame2, self.tracks_cam1, self.tracks_cam2, 
                                   cross_camera_matches)

                if output_path and video_writer is None:
                    h, w = vis.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, (w, h), True
                    )

                if video_writer is not None:
                    video_writer.write(vis)

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.namedWindow("Multi-Camera Tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Multi-Camera Tracking", vis)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frames_processed += 1
                skip_frames = 1  # Skip every other frame

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
        vis1, vis2 = frame1.copy(), frame2.copy()
        
        matched_ids = {t1_id: t2_id for t1_id, t2_id in matches 
                      if self.track_pairs[t1_id][t2_id] >= 3}

        for track_id, track in tracks1.items():
            if track.consecutive_hits >= self.config.min_hits:
                color = tuple(map(int, self.colors[track_id % len(self.colors)]))
                box = track.last_position.astype(np.int32)
                cv2.rectangle(vis1, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis1, str(track_id), (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for track_id, track in tracks2.items():
            if track.consecutive_hits >= self.config.min_hits:
                matching_id = next((t1_id for t1_id, t2_id in matched_ids.items() 
                                  if t2_id == track_id), track_id)
                color = tuple(map(int, self.colors[matching_id % len(self.colors)]))
                box = track.last_position.astype(np.int32)
                cv2.rectangle(vis2, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis2, str(matching_id), (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
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