
import cv2
import threading
import logging

logger = logging.getLogger(__name__)

class FrameReader(threading.Thread):
    """Asynchronous frame reader for video streams."""
    def __init__(self, video_path, queue, name=''):
        super().__init__()
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            raise ValueError(f"Cannot open video: {video_path}")
        self.queue = queue
        self.stopped = False
        self.name = name

    def run(self):
        try:
            while not self.stopped:
                ret, frame = self.video.read()
                if not ret:
                    self.stopped = True
                    self.queue.put(None)
                    break
                self.queue.put(frame)
        except Exception as e:
            logger.error(f"Error reading frames from {self.video_path}: {e}")
            self.stopped = True
            self.queue.put(None)

    def stop(self):
        self.stopped = True
        self.video.release()

