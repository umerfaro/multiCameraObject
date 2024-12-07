import numpy as np

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
