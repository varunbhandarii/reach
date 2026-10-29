"""Live camera, hand, perception and audio orchestration."""

from dataclasses import asdict
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from .audio import CueEngine
from .capture import Camera
from .controller import decide, target_center
from .hand import HandTracker
from .log import EventLog
from .overlay import draw_overlay
from .perception import Perception
from .trials import Trials
from .types import CONTRACT_VERSION


class Runner:
    def __init__(self, cfg, args, *, camera_factory=Camera, hand_factory=HandTracker,
                 audio_factory=CueEngine, perception_factory=Perception):
        self.cfg, self.args = cfg, args
        root = Path(args.log_dir or cfg.get("log", {}).get("dir", "runs"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_dir = root / stamp
        self.camera = camera_factory(cfg, log=None)
        self.hand = hand_factory(cfg)
        self.audio = audio_factory(cfg)
        self.log = EventLog(self.run_dir, save_frames=cfg.get("log", {}).get("save_frames", True))
        self.camera.log = self.log
        try:
            self.perception = perception_factory(cfg, self.camera, self.log)
        except Exception:
            self.log.close()
            raise
        self._stop = threading.Event()
        self._hand_thread = None
        self._latest_hand = self._decision = None
        self._hand_stamp = 0.0
        self._fatal = None
        self._muted = args.mute
        self._decision_count = 0
        self._started = None
        self._last_box = None
        self._latencies = deque(maxlen=18000)
        self._trial = (Trials(args.trials, args.trial_mode, cfg["objects"], self.log,
            seed=args.trial_seed, limit=args.trial_seconds) if args.trials else None)

    def _manifest(self):
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        config_hash = sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        return {"app_commit": result.stdout.strip() if result.returncode == 0 else "unknown",
                "config_hash": config_hash, "contract_version": CONTRACT_VERSION,
                "models": {"hand": "hand_landmarker.task", "sam": self.cfg.get("sam", {}).get("model")},
                "mount": self.args.mount, "os": platform.platform(), "source": self.camera.source,
                "prediction": self.args.predict, "muted": self._muted,
                "flip": self.args.flip, "hand_max_hands": self.cfg.get("hand", {}).get("max_hands", 2),
                "hand_input_width": self.cfg.get("hand", {}).get("input_width"),
                "motion_width": self.cfg.get("motion", {}).get("width", 640),
                "opencv_threads": self.cfg.get("runtime", {}).get("opencv_threads", 1),
                "objects": self.cfg["objects"], "trial_mode": self.args.trial_mode if self._trial else None,
                "trial_seed": self.args.trial_seed if self._trial else None,
                "camera_median_assumption_ms": self.args.camera_latency_ms,
                "latency_note": "Component estimate uses an earlier camera median; not an end-to-end latency measurement."}

    def _log_error(self, where, kind, message):
        try:
            self.log.emit("error", {"where": where, "kind": kind, "message": message})
        except Exception:
            self._fatal = "event writer failed; run data may be incomplete"

    def _hand_loop(self):
        last = -1
        previous_audible = False
        seq = 0
        try:
            while not self._stop.is_set():
                value = self.camera.wait_newer(last, timeout=.05)
                if value is None:
                    if time.perf_counter()-self._hand_stamp > .5:
                        self.audio.set(None)
                    if self.camera.stats().get("eof"):
                        self._stop.wait(.02)
                    continue
                frame, image = value
                last = frame.id
                target = self.perception.latest()
                center = target_center(target)
                hand = self.hand.observe(frame, image, center)
                now = time.perf_counter()
                self._latest_hand, self._hand_stamp = hand, now
                self.perception.on_hand(hand)
                flipped = self.args.flip == "on" or (self.args.flip == "auto" and self.hand.upside_down)
                # A blocked or delayed detector must not emit old guidance.
                target = self.perception.latest()
                current = target if now-frame.t_mono <= .5 else None
                decision = decide(hand, current, frame, seq=seq, prediction=self.args.predict,
                    flipped=flipped, gain=float(self.cfg.get("audio", {}).get("gain", .2)), now=now)
                audible = decision.cue is not None
                if audible != previous_audible and not self._muted:
                    self.audio.earcon("found" if audible else "lost")
                previous_audible = audible
                self.audio.set(None if self._muted else decision.cue)
                self._decision = decision
                self._decision_count += 1
                self.log.emit("hand", {"frame_id": frame.id, "present": hand.present,
                    "fingertip_px": hand.fingertip_px, "fingertip_raw_px": self.hand.fingertip_raw_px,
                    "handedness": hand.handedness, "score": hand.score,
                    "detect_ms": self.hand.last_detect_ms, "held": self.hand.held})
                self.log.emit("decision", asdict(decision))
                center = target_center(current)
                if hand.present and hand.fingertip_px and center:
                    self.log.emit("reach_sample", {"frame_id": frame.id,
                        "relative_xy": [(hand.fingertip_px[0]-center[0])/frame.width,
                                        (hand.fingertip_px[1]-center[1])/frame.height]})
                # Actual device output latency, not the acoustic round-trip probe.
                audio = self.audio.stats()
                output = audio.get("reported_output_latency_ms")
                if self.args.camera_latency_ms is not None and output is not None:
                    block = audio["device"]["blocksize"]/audio["device"]["samplerate"]*1000
                    latency = self.args.camera_latency_ms+(now-frame.t_mono)*1000+block/2+output
                    self._latencies.append(latency)
                seq += 1
        except Exception as exc:
            self.audio.set(None)
            self._fatal = f"hand/controller stopped ({type(exc).__name__})"
            try:
                self._log_error("hand_controller", type(exc).__name__, self._fatal)
            finally:
                self._stop.set()

    def _select_phrase(self, phrase):
        self.audio.set(None)
        self.perception.set_phrase(phrase)
        self.log.emit("ui", {"phrase": phrase})

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not (self._trial and self._trial.started is not None):
            self.audio.set(None)
            self.perception.select_point(x, y)
            self.log.emit("ui", {"click": [x, y]})

    def _key(self, key, frame):
        if key < 0:
            return
        key &= 255
        self.log.emit("ui", {"key": chr(key)})
        if key == ord("q"):
            self._stop.set()
        elif self._trial and key == ord(" "):
            self._trial.start(target_center(self.perception.latest()) is not None)
        elif self._trial and key == ord("t"):
            self._trial.finish(True, "reported_touch", self.perception)
        elif self._trial and key == ord("f"):
            self._trial.finish(False, "reported_failure", self.perception)
        elif self._trial and key == ord("n"):
            self._trial.prepare(self.perception, frame)
        elif key == ord("m"):
            self._muted = not self._muted
            self.audio.set(None)
            if not self._muted and not self.audio.running:
                self.audio.start()
        elif self._trial and self._trial.started is not None:
            return
        elif key == ord("c"):
            self._select_phrase(None)
        elif key == ord("r"):
            self.perception.force_query()
        elif ord("1") <= key < ord("1")+len(self.cfg["objects"]):
            self._select_phrase(self.cfg["objects"][key-ord("1")])

    def _metrics(self):
        elapsed = max(.001, time.perf_counter()-(self._started or time.perf_counter()))
        hand, camera, audio = self.hand.stats(), self.camera.stats(), self.audio.stats()
        latency = {"p50": float(np.median(self._latencies)), "p95": float(np.percentile(self._latencies, 95))} if self._latencies else None
        return {"duration_s": elapsed, "capture": camera, "hand": hand,
                "perception": self.perception.stats(), "audio": audio,
                "controller_hz": self._decision_count/elapsed, "motion_to_sound_estimate_ms": latency,
                "audio_measured": self.audio.running, "error": self._fatal,
                "trials": self._trial.results if self._trial else []}

    def run(self):
        window = "Reach"
        previous_switch = sys.getswitchinterval()
        previous_cv_threads = cv2.getNumThreads()
        sys.setswitchinterval(.001)
        cv2.setNumThreads(int(self.cfg.get("runtime", {}).get("opencv_threads", 1)))
        exit_code = 0
        metrics = None
        try:
            manifest = self._manifest()
            self.log.write_manifest(manifest)
            self.log.emit("session_start", manifest)
            self.camera.open()
            self.camera.start()
            if not self.args.mute:
                self.audio.start()
            self.perception.start()
            if self.args.phrase:
                self._select_phrase(self.args.phrase)
            self._started = time.perf_counter()
            self._hand_thread = threading.Thread(target=self._hand_loop, name="hand-controller", daemon=True)
            self._hand_thread.start()
            if not self.args.headless:
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(window, self._mouse)
            last_metrics = 0.0
            last_draw = 0.0
            initialized = False
            while not self._stop.is_set():
                now = time.perf_counter()
                camera = self.camera.stats()
                if camera["error"]:
                    raise RuntimeError(camera["error"])
                if self._fatal or self.audio.stats().get("error"):
                    raise RuntimeError("A processing or audio worker failed")
                if self.perception.stats().get("error"):
                    raise RuntimeError("Perception stopped; inspect the local run")
                if self.args.duration and now-self._started >= self.args.duration:
                    break
                value = self.camera.latest()
                if value is None:
                    if now-self._started > 5 or camera["eof"]:
                        raise RuntimeError("No usable frames received")
                    self._stop.wait(.01)
                    continue
                frame, image = value
                if not initialized:
                    if self.args.click:
                        self.perception.select_point(*self.args.click)
                    if self._trial:
                        self._trial.prepare(self.perception, frame)
                    initialized = True
                if now-frame.t_mono > .5 or now-self._hand_stamp > .5:
                    self.audio.set(None)
                if camera["eof"] and now-frame.t_mono > .2:
                    break
                target = self.perception.latest()
                if target and target_center(target):
                    self._last_box = target.instances[target.chosen].box_xyxy
                if self._trial:
                    self._trial.tick(self.perception)
                if now-last_metrics >= 1:
                    metrics = self._metrics()
                    self.log.emit("audio", metrics["audio"])
                    self.log.emit("loop_metrics", metrics)
                    print(f"fps {camera['capture_fps']:.1f}; hand p95 {metrics['hand']['hand_ms_p95']}; "
                          f"underflows {metrics['audio']['underflows']}", flush=True)
                    last_metrics = now
                if not self.args.headless and now-last_draw >= 1/30:
                    stats = {**camera, **self.perception.stats(), **(metrics["hand"] if metrics else {})}
                    canvas = draw_overlay(image, self._latest_hand, target, self._decision, stats,
                        muted=self._muted, trial=self._trial.lines() if self._trial else None, stale_box=self._last_box)
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) if sys.platform == "darwin" else cv2.pollKey()
                    self._key(key, frame)
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    last_draw = now
                self._stop.wait(.005 if self.args.headless else .001)
            if self._fatal:
                exit_code = 1
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            self._fatal = f"run stopped ({type(exc).__name__})"
            self._log_error("runner", type(exc).__name__, self._fatal)
            print(self._fatal, file=sys.stderr)
            exit_code = 1
        finally:
            self._stop.set()
            self.audio.set(None)
            if self._hand_thread:
                self._hand_thread.join(timeout=5)
                if self._hand_thread.is_alive():
                    self._fatal, exit_code = "hand worker did not stop", 1
            if self._trial and self._trial.started is not None:
                self._trial.finish(False, "interrupted", self.perception)
            for name, close in (("perception", self.perception.stop), ("camera", self.camera.stop),
                                ("hand", self.hand.close)):
                try:
                    close()
                except Exception:
                    self._fatal, exit_code = f"{name} cleanup failed", 1
                    self._log_error(name, "cleanup", self._fatal)
            metrics = self._metrics()
            metrics["completed"] = exit_code == 0
            try:
                self.log.emit("session_end", metrics)
            except Exception:
                metrics["completed"], metrics["error"], exit_code = False, "event writer failed", 1
            try:
                self.audio.stop()
            except Exception:
                metrics["completed"], metrics["error"], exit_code = False, "audio cleanup failed", 1
            try:
                try:
                    self.log.close()
                except Exception:
                    metrics["completed"], metrics["error"], exit_code = False, "event writer failed", 1
            finally:
                sys.setswitchinterval(previous_switch)
                cv2.setNumThreads(previous_cv_threads)
                if not self.args.headless:
                    cv2.destroyAllWindows()
            (self.run_dir/"summary.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            print(f"Run saved: {self.run_dir}", flush=True)
        return exit_code
