"""Paced file playback and continuously drained live camera capture."""

from collections import deque
from hashlib import sha256
from pathlib import Path
import math
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import cv2

from .types import Frame


class Camera:
    """Open on the main thread, then publish immutable frames to many readers."""

    def __init__(self, cfg, *, capture_factory=None, log=None):
        self.cfg = cfg.get("camera", {})
        self.log = log
        self._factory = capture_factory or cv2.VideoCapture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frames = deque(maxlen=int(self.cfg.get("ring_depth", 90)))
        self._cap = self._thread = None
        self._error = None
        self._stalls = self._recoveries = 0
        self._eof = False
        self._received = self._duplicates = self._published = 0
        self._started = self._finished = None
        self._source, self.source, self._backend = self._settings()

    def _settings(self):
        kind = self.cfg.get("source", "android_mjpeg")
        if kind == "file":
            value = self.cfg.get("path")
            if not value or not Path(value).is_file():
                raise ValueError("Set camera.path to an existing video file")
            value = str(value)
        elif kind == "android_mjpeg":
            value = self.cfg.get("url", "http://127.0.0.1:8080/video")
        elif kind in {"iphone_continuity", "usb_webcam"}:
            value = int(self.cfg.get("device_index", 0))
        else:
            raise ValueError("Unsupported camera source")
        backend = self.cfg.get("backend", "auto")
        if backend == "auto":
            backend = ("ffmpeg" if kind in {"file", "android_mjpeg"} else
                       "avfoundation" if sys.platform == "darwin" else
                       "dshow" if sys.platform == "win32" else "auto")
        backends = {"auto": cv2.CAP_ANY, "ffmpeg": cv2.CAP_FFMPEG,
                    "avfoundation": cv2.CAP_AVFOUNDATION, "dshow": cv2.CAP_DSHOW,
                    "msmf": cv2.CAP_MSMF}
        if backend not in backends:
            raise ValueError("Unsupported camera backend")
        return value, kind, backends[backend]

    def open(self):
        if self._cap is not None:
            return
        if sys.platform == "darwin" and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Open the camera from the main thread on macOS")
        if self.source == "android_mjpeg" and self.cfg.get("pin_landscape", False):
            parts = urlsplit(self._source)
            url = f"{parts.scheme}://{parts.netloc}/settings/orientation?set=landscape"
            try:
                with urlopen(url, timeout=2) as response:
                    response.read(256)
            except OSError:
                raise RuntimeError("Could not pin orientation; set landscape in the phone app") from None
        cap = self._create()
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Cannot open camera. Check source, permission and backend")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(self._source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("width", 640)))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("height", 480)))
            cap.set(cv2.CAP_PROP_FPS, 30)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if math.isfinite(fps) and fps > 0 else 30.0
        self._cap = cap

    def _create(self):
        if self._backend == cv2.CAP_FFMPEG:
            return self._factory(self._source, self._backend, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000])
        return self._factory(self._source, self._backend)

    def _event(self, event, **data):
        if self.log is not None:
            self.log.emit("camera", {"event": event, **data})

    def _reopen(self):
        """Reconnect a network stream after a stall; the next read shows whether it worked."""
        self._cap.release()
        self._cap = self._create()
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._event("reopen", opened=bool(self._cap.isOpened()))

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Camera may only be started once")
        self.open()
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._read, name="capture", daemon=True)
        self._thread.start()

    def _read(self):
        previous = None
        index = failures = 0
        outage = None
        limit = float(self.cfg.get("reconnect_s", 15))
        try:
            while not self._stop.is_set():
                if self.source == "file":
                    delay = self._started + index / self._fps - time.perf_counter()
                    if delay > 0 and self._stop.wait(delay):
                        break
                ok, image = self._cap.read()
                captured, wall = time.perf_counter(), time.time()
                if not ok:
                    if self.source == "file":
                        self._eof = True
                        break
                    failures += 1
                    if failures >= 3:
                        if outage is None:
                            outage = captured
                            self._stalls += 1
                            self._event("stalled")
                        # A phone stream over Wi-Fi can pause; reconnect before giving up.
                        if self.source == "android_mjpeg" and captured-outage < limit and not self._stop.is_set():
                            self._reopen()
                            failures = 0
                            self._stop.wait(.5)
                            continue
                        raise RuntimeError(f"Camera stopped delivering frames for {captured-outage:.1f} s")
                    self._stop.wait(.05)
                    continue
                if outage is not None:
                    self._recoveries += 1
                    self._event("recovered", outage_ms=round((captured-outage)*1000))
                    outage = None
                index += 1
                failures = 0
                self._received += 1
                encoded, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not encoded:
                    raise RuntimeError("Camera JPEG encoding failed")
                jpeg = jpeg.tobytes()
                digest = sha256(jpeg).hexdigest()
                if digest == previous:
                    self._duplicates += 1
                    continue
                previous = digest
                height, width = image.shape[:2]
                frame = Frame(self._published, captured, wall, width, height, digest, self.source)
                image.setflags(write=False)
                with self._condition:
                    self._frames.append((frame, image, jpeg))
                    self._published += 1
                    self._condition.notify_all()
                if self.log is not None:
                    self.log.emit("frame", {"frame_id": frame.id, "t_cap": captured,
                        "jpeg_sha256": digest, "width": width, "height": height,
                        "duplicates_dropped": self._duplicates})
                    self.log.save_frame(frame.id, jpeg)
        except Exception as exc:
            self._error = f"capture failed ({type(exc).__name__}: {str(exc)[:120]})"
        finally:
            self._cap.release()
            self._finished = time.perf_counter()
            with self._condition:
                self._condition.notify_all()

    def latest(self):
        with self._condition:
            return self._frames[-1][:2] if self._frames else None

    def wait_newer(self, last_id, timeout=.1):
        with self._condition:
            self._condition.wait_for(lambda: (self._frames and self._frames[-1][0].id > last_id)
                or self._finished is not None or self._stop.is_set(), timeout=max(0, timeout))
            if self._frames and self._frames[-1][0].id > last_id:
                return self._frames[-1][:2]
            return None

    def get(self, frame_id):
        with self._condition:
            return next((image for frame, image, _ in self._frames if frame.id == frame_id), None)

    def get_jpeg(self, frame_id):
        with self._condition:
            return next((jpeg for frame, _, jpeg in self._frames if frame.id == frame_id), None)

    def stats(self):
        elapsed = max(.001, (self._finished or time.perf_counter()) - (self._started or time.perf_counter()))
        return {"received": self._received, "published": self._published,
                "duplicates_dropped": self._duplicates,
                "duplicate_rate": self._duplicates / max(1, self._received),
                "capture_fps": self._received / elapsed, "published_fps": self._published / elapsed,
                "native_fps": getattr(self, "_fps", None), "elapsed_s": elapsed,
                "eof": self._eof, "error": self._error, "stalls": self._stalls, "recoveries": self._recoveries}

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=4)
            if self._thread.is_alive():
                raise RuntimeError("Camera driver did not stop within its deadline")
        elif self._cap is not None:
            self._cap.release()
