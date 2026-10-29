from hashlib import sha256
import threading
import time

import cv2
import numpy as np

from reach.capture import Camera


class FakeCapture:
    def __init__(self, images, fps=30):
        self.images = iter(images)
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        image = next(self.images, None)
        return image is not None, image

    def set(self, *args):
        pass

    def get(self, key):
        return self.fps

    def release(self):
        self.released = True


def test_pacing_duplicates_hash_and_multiple_readers(tmp_path):
    path = tmp_path / "clip.mp4"
    path.touch()
    images = [np.full((24, 32, 3), x, np.uint8) for x in [10, 10, 30, 60]]
    cap = FakeCapture(images, fps=20)
    camera = Camera({"camera": {"source": "file", "path": str(path)}}, capture_factory=lambda *args: cap)
    camera.open()
    started = time.perf_counter()
    camera.start()
    first = camera.wait_newer(-1, 1)
    assert first is not None
    frame, image = first
    assert not image.flags.writeable
    assert sha256(camera.get_jpeg(frame.id)).hexdigest() == frame.jpeg_sha256
    results = []
    threads = [threading.Thread(target=lambda: results.append(camera.wait_newer(frame.id, 1))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 2 and all(value[0].id > frame.id for value in results)
    while not camera.stats()["eof"]:
        time.sleep(.005)
    assert time.perf_counter() - started >= .18
    assert camera.stats()["duplicates_dropped"] == 1
    assert camera.stats()["published"] == 3
    assert camera.wait_newer(2, .01) is None
    camera.stop()
    assert cap.released


def test_ring_eviction_and_live_backend_timeouts():
    args_seen = []
    cap = FakeCapture([np.full((10, 10, 3), i, np.uint8) for i in range(6)])
    def factory(*args):
        args_seen.append(args)
        return cap
    camera = Camera({"camera": {"source": "android_mjpeg", "ring_depth": 2}}, capture_factory=factory)
    camera.start()
    camera._thread.join(2)
    assert camera.get(0) is None and camera.get(5) is not None
    assert args_seen[0][1] == cv2.CAP_FFMPEG
    assert args_seen[0][2] == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000]
    assert camera.stats()["error"]
    camera.stop()
