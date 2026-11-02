"""Live camera, hand, perception and audio orchestration."""

from dataclasses import asdict, replace
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
from .controller import target_center
from .hand import HandTracker
from .log import EventLog
from .overlay import draw_overlay
from .perception import Perception
from .trials import Trials
from .types import CONTRACT_VERSION
from .voice import VoiceClient
from .tts import Speaker, SilentSpeaker
from .session_io import CommandInbox, TrialActions, make_command
from .trial_resume import load_trial_resume, saved_trial_prefix
from .session_codec import replay_config


class Runner:
    def __init__(self, cfg, args, *, camera_factory=Camera, hand_factory=HandTracker,
                 audio_factory=CueEngine, perception_factory=Perception, brain_factory=None,
                 voice_factory=VoiceClient, speaker_factory=Speaker):
        self.cfg, self.args = cfg, args
        self._resume = None
        retained = []
        if args.resume_trials:
            self._resume = load_trial_resume(args.resume_trials, count=args.trials,
                mode=args.trial_mode, objects=cfg['objects'], seed=args.trial_seed,
                limit=args.trial_seconds, mount=args.mount, prediction=args.predict,
                source=cfg['camera']['source'])
            retained, _ = saved_trial_prefix(self._resume)
        if brain_factory is None:
            from .brain import Brain
            brain_factory = Brain
        self.brain = brain_factory(cfg, cfg['objects'])
        root = Path(args.log_dir or cfg.get("log", {}).get("dir", "runs"))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_dir = root / stamp
        self.camera = camera_factory(cfg, log=None)
        self.hand = hand_factory(cfg)
        self.audio = audio_factory(cfg)
        self.log = EventLog(self.run_dir, save_frames=cfg.get("log", {}).get("save_frames", True))
        self.camera.log = self.log
        self.inbox = CommandInbox(self.log)
        self.speaker = (speaker_factory if cfg.get('tts', {}).get('enabled', True) and not args.mute else SilentSpeaker)(cfg, self.audio, self.log)
        self.voice = (voice_factory(cfg, self.log, self.post_command,
            on_speech_start=lambda: self.speaker.cancel(info_only=True), speaking=lambda: self.speaker.busy,
            output_probe=self._output_probe) if cfg.get('voice', {}).get('enabled', False) else None)
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
        self._active_finished = None
        self._last_box = None
        self._latencies = deque(maxlen=18000)
        self._last_frame = None
        self._pending_click = None
        self._click_mode = False
        self._contact_signals = {}
        self._trial_actions = TrialActions(self)
        self._trial = (Trials(args.trials, args.trial_mode, cfg["objects"], self.log,
            seed=args.trial_seed, limit=args.trial_seconds, retained=retained) if args.trials else None)

    def _manifest(self):
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        config_hash = sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        return {"app_commit": result.stdout.strip() if result.returncode == 0 else "unknown",
                "config_hash": config_hash, "contract_version": CONTRACT_VERSION,
                "config": replay_config(self.cfg),
                "models": {"hand": "hand_landmarker.task", "sam": self.cfg.get("sam", {}).get("model")},
                "mount": self.args.mount, "os": platform.platform(), "source": self.camera.source,
                "prediction": self.args.predict, "muted": self._muted,
                "flip": self.args.flip, "hand_max_hands": self.cfg.get("hand", {}).get("max_hands", 2),
                "hand_input_width": self.cfg.get("hand", {}).get("input_width"),
                "motion_width": self.cfg.get("motion", {}).get("width", 640),
                "opencv_threads": self.cfg.get("runtime", {}).get("opencv_threads", 1),
                "objects": self.cfg["objects"], "trial_mode": self.args.trial_mode if self._trial else None,
                "trial_seed": self.args.trial_seed if self._trial else None,
                "trial_count": self.args.trials if self._trial else None,
                "trial_seconds": self.args.trial_seconds if self._trial else None,
                "trial_resume": self._resume,
                "camera_median_assumption_ms": self.args.camera_latency_ms,
                "voice": {k:v for k,v in self.cfg.get('voice', {}).items() if k not in ('api_key', 'access_token')},
                "tts": dict(self.cfg.get('tts', {})), "contact": dict(self.cfg.get('contact', {})),
                "speech_probe_interval_s": self.args.speech_every,
                "input_device": self.cfg.get('voice', {}).get('input_device'),
                "latency_note": "Component estimate uses an earlier camera median; not an end-to-end latency measurement."}

    def _log_error(self, where, kind, message):
        try:
            self.log.emit("error", {"where": where, "kind": kind, "message": message})
        except Exception:
            self._fatal = "event writer failed; run data may be incomplete"

    def _output_probe(self):
        from .audio import select_output
        return select_output(self.cfg.get('audio', {}))[2]

    def post_command(self, command):
        trial = self._trial
        if (trial and trial.prepared and command.kind == 'find'
                and command.phrase != trial.phrase):
            self.log.emit('command', {**asdict(command), 'ignored_for_trial': True,
                                     'reason': 'scripted target is authoritative'})
            return
        self.inbox.post(command)

    def select_point(self, x, y):
        self._pending_click = (x, y)
        self.post_command(make_command('find', 'clicked point', source='trial' if self._trial else 'key'))

    def _effects(self, effects, now):
        for effect in effects:
            if effect.kind == 'speak':
                if not (self._trial and self._trial.prepared and self._trial.started is None and effect.key == 'hand_lost'):
                    self.speaker.say(effect.text, key=effect.key, priority=effect.priority)
            elif effect.kind == 'cancel_speech':
                self.speaker.cancel()
            elif effect.kind == 'earcon':
                if not self._muted:
                    self.audio.earcon(effect.key)
                if effect.key == 'contact' and self._trial:
                    self._trial.note_contact(now)
            elif effect.kind == 'set_phrase':
                if self._click_mode and effect.text == 'clicked point' and self._pending_click is not None:
                    self.perception.select_point(*self._pending_click)
                    self._pending_click = None
                else:
                    self.perception.set_phrase(effect.text)
            elif effect.kind == 'force_query':
                if self._click_mode and self._pending_click is not None:
                    self.perception.select_point(*self._pending_click)
                    self._pending_click = None
                else:
                    self.perception.force_query()
            elif effect.kind == 'choose_instance':
                self.perception.choose(effect.payload['sam_id'])
            elif effect.kind == 'exclude_instance':
                self.perception.exclude(effect.payload['frame_id'],effect.payload['box_xyxy'])
            elif effect.kind == 'perception_state':
                self.perception.set_state(effect.key)
            elif effect.kind == 'log':
                payload = dict(effect.payload or {})
                kind = effect.key or payload.pop('type', None) or payload.pop('kind', None)
                if kind is None:
                    kind = 'state' if 'from' in payload and 'to' in payload else 'contact' if 'inside' in payload else 'session'
                data = payload.get('data', payload.get('payload', payload))
                self.log.emit(kind, data)
                if kind == 'contact':
                    self._contact_signals = dict(data)

    def _session_step(self, now, hand=None, target=None, flipped=None):
        self.speaker.poll()
        commands = self.inbox.drain()
        for command in commands:
            if command.kind == 'find':
                self._click_mode = command.phrase == 'clicked point' and self._pending_click is not None
            elif command.kind in ('stop', 'cancel', 'start_over'):
                self._click_mode = False
        frame = self._last_frame
        if hand is None:
            hand = self._latest_hand if now-self._hand_stamp <= .5 else None
        if target is None:
            target = self.perception.latest()
        if frame is None or now-frame.t_mono > .5:
            hand, target = None, None
        if self._click_mode and target is not None:
            target = replace(target, phrase='clicked point')
        if self._trial and self._trial.prepared and self._trial.started is None:
            hand = None  # Contact is disarmed until the timed trial starts.
        stats = self.perception.stats()
        voice = self.voice.stats() if self.voice else {'connected': True, 'stereo_ok': True}
        stats.update(voice_connected=voice['connected'], voice_online=voice['connected'], voice_offline=not voice['connected'])
        if flipped is None:
            flipped = self.args.flip == 'on' or (self.args.flip == 'auto' and self.hand.upside_down)
        decision, effects = self.brain.step(now, hand=hand, target=target, perception_stats=stats,
            commands=commands, speaking=self.speaker.busy,
            frame_size=(frame.width, frame.height) if frame else (640, 480), flipped=flipped)
        self._effects(effects, now)
        self.audio.set(None if self._muted or not voice.get('stereo_ok', True) else decision.cue)
        self._decision = decision
        self._decision_count += 1
        self.log.emit('decision', asdict(decision))

    def _hand_loop(self):
        last = -1
        try:
            while not self._stop.is_set():
                value = self.camera.wait_newer(last, timeout=.05)
                if self._stop.is_set():
                    break
                if value is None:
                    self._session_step(time.perf_counter())
                    if time.perf_counter()-self._hand_stamp > .5:
                        self.audio.set(None)
                    if self.camera.stats().get("eof"):
                        self._stop.wait(.02)
                    continue
                frame, image = value
                self._last_frame = frame
                last = frame.id
                target = self.perception.latest()
                center = target_center(target)
                hand = self.hand.observe(frame, image, center)
                if self._stop.is_set():
                    break
                now = time.perf_counter()
                self._latest_hand, self._hand_stamp = hand, now
                self.perception.on_hand(hand)
                flipped = self.args.flip == "on" or (self.args.flip == "auto" and self.hand.upside_down)
                # A blocked or delayed detector must not emit old guidance.
                target = self.perception.latest()
                current = target if now-frame.t_mono <= .5 else None
                self._session_step(now, hand=hand, target=current, flipped=flipped)
                self.log.emit("hand", {"frame_id": frame.id, "present": hand.present,
                    "fingertip_px": hand.fingertip_px, "fingertip_raw_px": self.hand.fingertip_raw_px,
                    "handedness": hand.handedness, "score": hand.score,
                    "landmarks_px": hand.landmarks_px,
                    "detect_ms": self.hand.last_detect_ms, "held": self.hand.held})
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
        except Exception as exc:
            self.audio.set(None)
            self._fatal = f"hand/controller stopped ({type(exc).__name__})"
            try:
                self._log_error("hand_controller", type(exc).__name__, self._fatal)
            finally:
                self._stop.set()

    def _select_phrase(self, phrase):
        self.audio.set(None)
        self.post_command(make_command('find' if phrase else 'cancel', phrase))
        self.log.emit("ui", {"phrase": phrase})

    def _mouse(self, event, x, y, flags, param):
        if self._trial and (self._trial.mode != "click" or self._trial.started is not None
                            or not self._trial.prepared):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.audio.set(None)
            self.select_point(x, y)
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
            self._trial.finish(True, "reported_touch", self._trial_actions)
        elif self._trial and key == ord("f"):
            self._trial.finish(False, "reported_failure", self._trial_actions)
        elif self._trial and key == ord("n"):
            self._trial.prepare(self._trial_actions, frame)
        elif key == ord("m"):
            self._muted = not self._muted
            self.audio.set(None)
            if hasattr(self.audio, 'set_muted'):
                self.audio.set_muted(self._muted)
            if not self._muted and not self.audio.running:
                self.audio.start()
        elif key == ord('c'):
            self._select_phrase(None)
        elif key == ord('v') and self.voice:
            self.voice.toggle_ptt()
        elif self._trial:
            if key == ord("r"):
                self.post_command(make_command('recheck'))
            return
        elif key == ord("r"):
            self.post_command(make_command('recheck'))
        elif key in (ord('y'), ord('n')):
            self.post_command(make_command('yes' if key == ord('y') else 'no'))
        elif ord("1") <= key < ord("1")+len(self.cfg["objects"]):
            self._select_phrase(self.cfg["objects"][key-ord("1")])

    def _metrics(self):
        elapsed = max(.001, (self._active_finished or time.perf_counter())-(self._started or time.perf_counter()))
        hand, camera, audio = self.hand.stats(), self.camera.stats(), self.audio.stats()
        latency = {"p50": float(np.median(self._latencies)), "p95": float(np.percentile(self._latencies, 95))} if self._latencies else None
        return {"duration_s": elapsed, "capture": camera, "hand": hand,
                "voice": self.voice.stats() if self.voice else {'enabled': False},
                "speech": self.speaker.stats(), "contact": self._contact_signals,
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
            if self._resume:
                self.log.emit('trial_resume', self._resume)
                print(f"Continuing at trial {self._resume['interrupted_trial']}/{self.args.trials}; "
                      f"{self._resume['completed_prefix']} earlier outcomes retained. "
                      "The interrupted attempt remains in the original run.", flush=True)
            self.camera.open()
            self.camera.start()
            if not self.args.mute:
                self.audio.start()
                self.speaker.start()
                print('Preparing local speech; wait for the camera window before speaking.', flush=True)
                self.speaker.wait_ready()
            self.perception.start()
            if self.voice:
                self.voice.start()
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
            next_speech = time.perf_counter()+self.args.speech_every if self.args.speech_every else float('inf')
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
                if now >= next_speech:
                    self.speaker.say('Ready.', key='ready', priority=0)
                    next_speech = now+self.args.speech_every
                value = self.camera.latest()
                if value is None:
                    if now-self._started > 5 or camera["eof"]:
                        raise RuntimeError("No usable frames received")
                    self._stop.wait(.01)
                    continue
                frame, image = value
                if not initialized:
                    if self.args.click:
                        self.select_point(*self.args.click)
                    if self._trial:
                        self._trial.prepare(self._trial_actions, frame)
                    initialized = True
                if now-frame.t_mono > .5 or now-self._hand_stamp > .5:
                    self.audio.set(None)
                if camera["eof"] and now-frame.t_mono > .2:
                    break
                target = self.perception.latest()
                if target and target_center(target):
                    self._last_box = target.instances[target.chosen].box_xyxy
                if self._trial:
                    self._trial.tick(self._trial_actions)
                if now-last_metrics >= 1:
                    metrics = self._metrics()
                    self.log.emit("audio", metrics["audio"])
                    self.log.emit('voice_status', metrics['voice'])
                    self.log.emit("loop_metrics", metrics)
                    print(f"fps {camera['capture_fps']:.1f}; hand p95 {metrics['hand']['hand_ms_p95']}; "
                          f"underflows {metrics['audio']['underflows']}", flush=True)
                    last_metrics = now
                if not self.args.headless and now-last_draw >= 1/30:
                    stats = {**camera, **self.perception.stats(), **(metrics["hand"] if metrics else {})}
                    stats.update(voice=self.voice.stats() if self.voice else {'enabled': False},
                                 speech=self.speaker.stats(), contact=self._contact_signals)
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
            self._active_finished = time.perf_counter()
            self._stop.set()
            self.audio.set(None)
            if self._hand_thread:
                self._hand_thread.join(timeout=5)
                if self._hand_thread.is_alive():
                    self._fatal, exit_code = "hand worker did not stop", 1
            if self._trial and self._trial.started is not None:
                self._trial.finish(False, "interrupted", self._trial_actions)
            for name, close in (("perception", self.perception.stop), ("camera", self.camera.stop),
                                ("hand", self.hand.close)):
                try:
                    close()
                except Exception:
                    self._fatal, exit_code = f"{name} cleanup failed", 1
                    self._log_error(name, "cleanup", self._fatal)
            if self.voice:
                try:
                    self.voice.stop()
                except Exception:
                    self._fatal, exit_code = 'voice cleanup failed', 1
                    self._log_error('voice', 'cleanup', self._fatal)
            metrics = self._metrics()
            metrics["completed"] = exit_code == 0
            try:
                manifest['input_device'] = self.voice.stats().get('input_name') if self.voice else None
                manifest['active_start_mono'] = self._started
                manifest['active_end_mono'] = self._active_finished
                self.log.write_manifest(manifest)
                self.log.emit("session_end", metrics)
            except Exception:
                metrics["completed"], metrics["error"], exit_code = False, "event writer failed", 1
            try:
                self.speaker.poll()
                self.speaker.stop()
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
