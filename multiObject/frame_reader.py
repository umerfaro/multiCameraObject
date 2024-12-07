import cv2
import threading

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
