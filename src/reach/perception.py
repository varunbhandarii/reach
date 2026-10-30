"""Camera-motion perception with a single blocking segmentation worker."""

from dataclasses import asdict
from hashlib import sha256
import math
import queue
import threading
import time

import cv2

from .appearance import compare_appearance, appearance_changed
from .sam_client import SamClient, SamError, SamRateLimited
from .sam_cache import SamCache, CacheMiss
from .scheduler import QueryScheduler
from .tracker import TargetTracker, box_iou
from .types import Instance, TargetObs


class Perception:
    """Start after Camera.open/start; stop before closing the run log.

    Camera supplies wait_newer(last_id, timeout), returning (Frame, BGR image)
    or None, and get_jpeg(frame_id). No camera is opened by this component.
    """

    def __init__(self, cfg, camera, log, *, segmenter=None, clock=time.perf_counter):
        self.cfg, self.camera, self.log = cfg, camera, log
        self.tracker = TargetTracker(cfg)
        self.scheduler = QueryScheduler(cfg)
        self._segment = segmenter or SamClient(cfg).segment
        self._cache = SamCache(cfg)
        self._clock = clock
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._jobs, self._results = queue.Queue(maxsize=1), queue.SimpleQueue()
        self._thread = self._worker_thread = None
        self._target = self._hand = None
        self._mode = self._phrase = self._point = None
        self._generation = 0
        self._seen_generation = -1
        self._state = "IDLE"
        self._inflight = None
        self._last_applied = -1
        self._req_id = 0
        self._seed_time = None
        self._ambiguous = False
        self._metrics = {"images_sent": 0, "responses_dropped": 0, "errors": 0, "cache_hits": 0, "appearance_rejections": 0}
        self._fatal = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Perception may only be started once")
        self._worker_thread = threading.Thread(target=self._sam_worker, name="sam-worker", daemon=True)
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._worker_thread.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._generation += 1
            self._state = "STOPPED"
            self._target = None
        if self._thread:
            self._thread.join(timeout=4)
        if self._worker_thread:
            self._worker_thread.join(timeout=4)
        if (self._thread and self._thread.is_alive()) or (self._worker_thread and self._worker_thread.is_alive()):
            raise RuntimeError("perception worker did not stop within its deadline")
        with self._lock:
            # The worker may finish after the frame loop exits. Preserve its
            # terminal response before the runner closes the event log.
            self._drain_results()
            if self._inflight is not None:
                job = self._inflight
                self._emit("sam_cancel", {"req_id": job["id"], "frame_id": job["frame"].id,
                    "reason": "shutdown_before_send"})
                self._inflight = None
            while True:
                try:
                    self._jobs.get_nowait()
                except queue.Empty:
                    break

    def set_phrase(self, phrase: str | None):
        phrase = phrase.strip() if phrase else None
        if phrase and (len(phrase) > 120 or any(c in phrase for c in '\n,;')):
            raise ValueError("choose one short noun phrase")
        with self._lock:
            if (phrase and self._mode == "phrase" and phrase == self._phrase
                    and self._state not in ("IDLE", "STOPPED") and not self._fatal):
                self.scheduler.force()
                return
            self._generation += 1
            self._mode, self._phrase, self._point = ("phrase" if phrase else None), phrase, None
            self._target = None
            self._state = "SEARCHING" if phrase else "IDLE"
            self._fatal = None

    def select_point(self, x: float, y: float):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("point must be finite")
        with self._lock:
            self._generation += 1
            self._mode, self._phrase, self._point = "click", "", (x, y)
            self._target = None
            self._state = "GUIDING"

    def set_state(self, state: str):
        with self._lock:
            self._state = state.upper()
            if self._state in ("IDLE", "STOPPED"):
                self._generation += 1
                self._target = None

    def force_query(self):
        """Runner's recheck key; never overrides idle or one-in-flight guards."""
        with self._lock:
            self.scheduler.force()

    def on_hand(self, hand):
        self._hand = hand

    def latest(self) -> TargetObs | None:
        target = self._target
        frame = self.tracker.frame
        if frame is None or self._clock()-frame.t_mono > .5:
            return None
        return target

    def stats(self):
        with self._lock:
            return {**self._metrics, **self.tracker.stats(), "ambiguous": self._ambiguous,
                    "in_flight": self._inflight is not None, "state": self._state,
                    "error": self._fatal, "estimated_cost_usd": self._metrics["images_sent"]*.0025}

    def _emit(self, kind, data):
        if self.log is not None:
            self.log.emit(kind, data)

    def _sam_worker(self):
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=.05)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            started = self._clock()
            with self._lock:
                cancelled = job["generation"] != self._generation or self._state in ("IDLE", "STOPPED")
            if cancelled:
                self._results.put((job, (), None, 0.0))
                continue
            try:
                result = self._cache.read(job["jpeg"], job["phrase"], job["frame"])
                if result is not None:
                    with self._lock:
                        self._metrics["cache_hits"] += 1
                    self._emit("sam_cache", {"req_id": job["id"], "frame_id": job["frame"].id, "hit": True})
                else:
                    if self._cache.mode == "read":
                        raise CacheMiss("no cached segmentation; read mode does not send images")
                    with self._lock:
                        self._metrics["images_sent"] += 1
                    self._emit("sam_req", {"req_id": job["id"], "frame_id": job["frame"].id,
                        "phrase": job["phrase"], "jpeg_bytes": len(job["jpeg"])})
                    result = self._segment(job["jpeg"], job["phrase"], job["frame"])
                    self._cache.write(job["jpeg"], job["phrase"], job["frame"], result)
                error = None
            except Exception as exc:
                result = ()
                error = exc if isinstance(exc, SamError) else SamError("segmentation worker failed")
            self._results.put((job, result, error, (self._clock()-started)*1000))

    def _run(self):
        last = -1
        try:
            while not self._stop.is_set():
                value = self.camera.wait_newer(last, timeout=.05)
                if value is None:
                    with self._lock:
                        self._drain_results()
                    continue
                frame, image = value
                if frame.id <= last:
                    continue
                self.process(frame, image)
                last = frame.id
        except Exception:
            with self._lock:
                self._target = None
                self._fatal = "perception thread failed; inspect the local run"
                self._metrics["errors"] += 1
            self._emit("error", {"where": "perception", "kind": "worker", "message": self._fatal})
            self._stop.set()

    def _drain_results(self):
        while True:
            try:
                job, instances, error, latency = self._results.get_nowait()
            except queue.Empty:
                return
            self._inflight = None
            self._metrics["sam_latency_ms"] = latency
            status = "error" if error else "ok"
            if error:
                self._metrics["errors"] += 1
                if isinstance(error, SamRateLimited):
                    self.scheduler.backoff(self._clock(), error.retry_after)
                elif error.kind in ("permanent", "cache_miss"):
                    if job["generation"] == self._generation:
                        self._fatal = str(error)
                else:
                    self.scheduler.backoff(self._clock(), 1)
            invalid = (job["generation"] != self._generation or self._mode != "phrase"
                       or self._state in ("IDLE", "STOPPED") or job["frame"].id <= self._last_applied
                       or self._clock()-job["submitted"] > 3
                       or self.tracker.compose(job["frame"].id) is None)
            if not error and invalid:
                status = "dropped"
                self._metrics["responses_dropped"] += 1
            self._emit("sam_resp", {"req_id": job["id"], "frame_id": job["frame"].id,
                "status": status, "latency_ms": latency, "instances": [asdict(i) for i in instances],
                "error": error.kind if error else None,
                "reason": "shutdown" if self._stop.is_set() else None})
            if error or invalid:
                continue
            previous = self.tracker.current()
            candidates = [(i, self.tracker.map_instance(job["frame"].id, item)) for i, item in enumerate(instances)]
            candidates = [(i, item) for i, item in candidates if item is not None]
            if not candidates:
                # No detection is not permission to retain an unconfirmed target.
                self.tracker.clear()
                self._target = None
                self._state = "SEARCHING"
                continue
            confidence = lambda pair: pair[1].conf if pair[1].conf is not None else 0
            ranked = sorted(candidates, key=confidence, reverse=True)
            tie = float(self.cfg.get("sam", {}).get("tie_margin", .15))
            self._ambiguous = len(ranked) > 1 and confidence(ranked[0])-confidence(ranked[1]) <= tie
            if previous is not None:
                ranked.sort(key=lambda pair: box_iou(previous.box_xyxy, pair[1].box_xyxy), reverse=True)
            selected, mapped = ranked[0]
            motion_cfg = self.cfg.get("motion", {})
            get_image = getattr(self.camera, "get", None)
            source_image = get_image(job["frame"].id) if callable(get_image) else None
            if motion_cfg.get("appearance_guard", True) and source_image is not None:
                distances = compare_appearance(source_image, instances[selected], self.tracker.image, mapped, self._hand)
                if appearance_changed(distances):
                    self._metrics["appearance_rejections"] += 1
                    self.tracker.clear()
                    self._target = None
                    self._state = "SEARCHING"
                    self.scheduler.force()
                    self._emit("target_reject", {"req_id": job["id"], "frame_id": self.tracker.frame.id,
                        "source_frame_id": job["frame"].id, "reason": "appearance_changed", **distances})
                    continue
            if self.tracker.reseed(job["frame"].id, instances[selected]):
                self._last_applied = job["frame"].id
                self._seed_time = job["frame"].t_mono
                self._state = "GUIDING"
                self._publish("sam")

    def _publish(self, source="tracker"):
        current = self.tracker.current()
        if current is None or self.tracker.stale or self._state in ("IDLE", "STOPPED"):
            self._target = None
        else:
            age = max(0, (self._clock()-self._seed_time)*1000) if self._seed_time is not None else 0
            self._target = TargetObs(self.tracker.frame.id, self._phrase or "", (current,), 0, source, age)
        target = self._target
        self._emit("target", {"frame_id": self.tracker.frame.id, "source": target.source if target else "none",
            "chosen": target.chosen if target else None, "box_xyxy": current.box_xyxy if current else None,
            "stale": self.tracker.stale, "age_ms": target.age_ms if target else None, "ambiguous": self._ambiguous})

    def process(self, frame, image):
        """One frame step, also usable in deterministic file tests."""
        with self._lock:
            self.tracker.update(frame, image, self._hand)
            self._emit("motion", self.tracker.motion)
            if self._seen_generation != self._generation:
                self.tracker.clear()
                self.scheduler.reset()
                self._last_applied = -1
                self._ambiguous = False
                self._seen_generation = self._generation
                if self._mode == "click":
                    x, y = self._point
                    box = (max(0, int(x-40)), max(0, int(y-40)), min(frame.width, int(x+40)), min(frame.height, int(y+40)))
                    if box[2] > box[0] and box[3] > box[1]:
                        self.tracker.reseed(frame.id, Instance("click", box, 1.0, None))
                        self._seed_time = frame.t_mono
            self._drain_results()
            self._publish()
            if self._mode != "phrase" or self._fatal:
                return
            near = False
            target, hand = self._target, self._hand
            if target and hand and hand.present and hand.fingertip_px is not None:
                x0, y0, x1, y1 = target.instances[0].box_xyxy
                near = math.dist(hand.fingertip_px, ((x0+x1)/2, (y0+y1)/2)) <= 1.5*(x1-x0)
            now = self._clock()
            if not self.scheduler.should_query(self._state, target is not None, now, in_flight=self._inflight is not None, near=near):
                return
            jpeg = self.camera.get_jpeg(frame.id)
            if jpeg is None:
                return
            if sha256(jpeg).hexdigest() != frame.jpeg_sha256:
                self._emit("error", {"where": "perception", "kind": "frame_hash", "message": "stored JPEG does not match its frame"})
                return
            job = {"id": self._req_id, "frame": frame, "jpeg": jpeg, "phrase": self._phrase,
                   "generation": self._generation, "submitted": now}
            self._req_id += 1
            self._inflight = job
            self._jobs.put_nowait(job)
