import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    conf_threshold: float = 0.30
    cos_threshold: float = 0.80
    reid_model_name: str = "osnet_x0_25"  # Use lighter ReID model
    reid_weights_path: str = "./weights/osnet_x0_25.pth.tar"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (416,416)
    batch_size: int = 128  # Larger batch size for GPU utilization
    max_feature_distance: float = 0.7
    reid_keep_frames: int = 30  # Number of frames to keep ReID features

