"""Command line for live guidance, device discovery and saved-run summaries."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from .config import load_config


def parser():
    result = argparse.ArgumentParser(prog="reach")
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run camera, hand tracking and stereo guidance")
    run.add_argument("--config", type=Path)
    run.add_argument("--source", help="Camera URL, device index, or video path")
    run.add_argument("--mount", required=True, choices=["handheld", "chest", "fixed"])
    run.add_argument("--phrase")
    run.add_argument("--objects", help="Comma-separated nouns for keys 1 through 4")
    run.add_argument("--predict", type=float, choices=[0, .5], default=.5)
    run.add_argument("--mute", action="store_true", help="Do not open audio output")
    run.add_argument("--log-dir", type=Path)
    run.add_argument("--sam-cache", choices=["off", "read", "write"])
    run.add_argument("--headless", action="store_true")
    run.add_argument("--duration", type=float, default=0, help="Stop after this many seconds; 0 runs until quit")
    run.add_argument("--click", nargs=2, type=float, metavar=("X", "Y"))
    run.add_argument("--hand-width", type=int, help="Resize only hand inference, e.g. 480; coordinates remain source pixels")
    run.add_argument("--max-hands", type=int, choices=[1, 2], help="Optional measured single-hand performance fallback; default 2")
    run.add_argument("--motion-width", type=int, choices=[320, 640], help="Width of local scene-motion inference")
    run.add_argument("--flip", choices=["auto", "on", "off"], default="auto")
    run.add_argument("--camera-latency-ms", type=float, help="Previously measured camera median, for a component latency estimate")
    run.add_argument("--trials", type=int, default=0)
    run.add_argument("--trial-mode", choices=["click", "phrase"], default="phrase")
    run.add_argument("--trial-seed", type=int, default=1)
    run.add_argument("--trial-seconds", type=float, default=45)
    commands.add_parser("devices", help="List audio devices and host APIs")
    ears = commands.add_parser("earcons", help="Listen to each earcon on the configured stereo output")
    ears.add_argument("--config", type=Path)
    report = commands.add_parser("report", help="Summarize one saved run")
    report.add_argument("run_dir", type=Path)
    replay = commands.add_parser("replay-contact", help="Replay contact geometry from a saved local run")
    replay.add_argument("run_dir", type=Path)
    replay.add_argument("--frames", action="store_true")
    replay.add_argument("--force", action="store_true")
    return result


def configure(args):
    cfg = deepcopy(load_config(args.config))
    camera = cfg.setdefault("camera", {})
    if args.source is not None:
        source = args.source
        if source.isdigit():
            camera["device_index"] = int(source)
            camera["source"] = "iphone_continuity" if sys.platform == "darwin" else "usb_webcam"
        elif source.startswith(("http://", "https://")):
            camera["source"], camera["url"] = "android_mjpeg", source
        else:
            camera["source"], camera["path"] = "file", str(Path(source).expanduser().resolve())
        camera["backend"] = "auto"
    if args.sam_cache is not None:
        cfg.setdefault("sam", {})["cache"] = args.sam_cache
    cfg["objects"] = ([part.strip() for part in args.objects.split(",") if part.strip()] if args.objects else
                       cfg.get("objects", ["wallet", "pen", "ball", "apple"]))
    if not 1 <= len(cfg["objects"]) <= 4:
        raise ValueError("Choose one to four objects")
    if args.hand_width is not None:
        if args.hand_width < 160:
            raise ValueError("Hand input width must be at least 160")
        cfg.setdefault("hand", {})["input_width"] = args.hand_width
    if args.max_hands is not None:
        cfg.setdefault("hand", {})["max_hands"] = args.max_hands
    if args.motion_width is not None:
        cfg.setdefault("motion", {})["width"] = args.motion_width
    if args.duration < 0 or args.trials < 0 or args.trial_seconds <= 0:
        raise ValueError("Duration and trial counts must be nonnegative, trial time positive")
    if args.camera_latency_ms is not None and args.camera_latency_ms < 0:
        raise ValueError("Camera latency cannot be negative")
    if args.headless and args.trials:
        raise ValueError("Interactive trials need the overlay window")
    if args.phrase and args.click:
        raise ValueError("Choose either a phrase or a clicked point")
    if args.trials and (args.phrase or args.click):
        raise ValueError("Trial mode chooses its own target")
    return cfg


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        if args.command == "devices":
            import sounddevice as sd
            print(sd.query_hostapis())
            print(sd.query_devices())
            return 0
        if args.command == "earcons":
            from .audio import CueEngine, EARCONS
            engine = CueEngine(load_config(args.config))
            print("Use low system volume. Earcons play one at a time.", flush=True)
            try:
                engine.start()
                for name in EARCONS:
                    print(name, flush=True)
                    engine.earcon(name)
                    time.sleep(.8)
                if engine.stats()["error"]:
                    return 1
            finally:
                engine.stop()
            return 0
        if args.command == "replay-contact":
            from .replay_contact import replay_contact
            print(json.dumps(replay_contact(args.run_dir, frames=args.frames, force=args.force), indent=2))
            return 0
        if args.command == "report":
            from .session_report import summarize
            print(json.dumps(summarize(args.run_dir), indent=2))
            return 0
        cfg = configure(args)
        from .runner import Runner
        return Runner(cfg, args).run()
    except (ValueError, OSError, RuntimeError) as exc:
        # Device/config errors are actionable; never echo arbitrary network exception bodies.
        print(f"Cannot start Reach ({type(exc).__name__}). Check config, camera, hand model and audio device.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
