"""Command line for live guidance, device discovery and saved-run summaries."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from .config import load_config
from .trial_resume import TrialResumeError


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
    run.add_argument('--resume-trials', type=Path, help='Continue an interrupted scripted run in a new linked folder; --trials keeps the original total')
    run.add_argument("--trial-mode", choices=["click", "phrase", "voice"], default="phrase")
    run.add_argument('--no-voice', action='store_true', help='Do not open or upload microphone audio')
    run.add_argument('--voice', action='store_true', help='Explicitly enable microphone streaming with file or muted sources')
    run.add_argument('--voice-input', help='Microphone name on the native host API')
    run.add_argument('--speech', choices=['on', 'off'], default='on')
    run.add_argument('--contact', choices=['on', 'off'], default='on')
    run.add_argument('--readback', choices=['on','off'], default='on')
    run.add_argument('--spark-cache', choices=['off','read','write'])
    run.add_argument('--record', action='store_true', help='Save local overlay video and stereo output audio')
    run.add_argument('--record-format', choices=['mp4','avi'], default='mp4')
    run.add_argument('--overlay-scale',type=int,choices=[1,2],default=1)
    run.add_argument('--speech-every', type=float, default=0, help='Optional interval for a local speech load probe')
    run.add_argument("--trial-seed", type=int, default=1)
    run.add_argument("--trial-seconds", type=float, default=45)
    commands.add_parser("devices", help="List audio devices and host APIs")
    ears = commands.add_parser("earcons", help="Listen to each earcon on the configured stereo output")
    ears.add_argument("--config", type=Path)
    speak = commands.add_parser('speak', help='Render local speech through the configured cue output')
    speak.add_argument('text')
    speak.add_argument('--config', type=Path)
    speak.add_argument('--count', type=int, default=1)
    speak.add_argument('--log-dir', type=Path, default=Path('runs/speech'))
    report = commands.add_parser("report", help="Summarize one saved run")
    report.add_argument("run_dir", type=Path)
    report.add_argument('--html',type=Path,metavar='OUT.html',help='Write a self-contained event timeline')
    probe = commands.add_parser('readback-probe', help='Measure standard Spark on a saved crop (uploads that image)')
    probe.add_argument('image', type=Path)
    probe.add_argument('--reference',type=Path,help='Clean crop of the target before reaching')
    probe.add_argument('--phrase', required=True)
    probe.add_argument('--wrong-image', type=Path, help='Known wrong-object crop for the same requested phrase')
    probe.add_argument('--config', type=Path)
    probe.add_argument('--effort', choices=['minimal','low'], default='minimal')
    probe.add_argument('--detail', choices=['low','auto'], default='low')
    probe.add_argument('--count', type=int, default=1)
    probe.add_argument('--matrix', action='store_true', help='Measure both efforts and both detail levels')
    probe.add_argument('--count-tokens', action='store_true')
    probe.add_argument('--log-dir', type=Path, default=Path('runs/readback-probe'))
    replay = commands.add_parser("replay-contact", help="Replay contact geometry from a saved local run")
    replay.add_argument("run_dir", type=Path)
    replay.add_argument("--frames", action="store_true")
    replay.add_argument("--force", action="store_true")
    replay_full=commands.add_parser('replay', help='Verify saved controller decisions without devices or network')
    replay_full.add_argument('run_dir',type=Path)
    replay_full.add_argument('--strip',type=Path,metavar='OUTPUT_DIR',help='Create a fresh events-only copy with canonical command text')
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
    cfg.setdefault('voice', {})['enabled'] = (not args.no_voice and
        (args.voice or (camera.get('source') != 'file' and not args.mute)))
    if args.voice_input:
        cfg['voice']['input_device'] = args.voice_input
    cfg.setdefault('tts', {})['enabled'] = args.speech == 'on' and not args.mute
    cfg.setdefault('contact', {})['enabled'] = args.contact == 'on'
    cfg.setdefault('brain', {})['prediction'] = args.predict
    cfg['brain']['readback_enabled'] = args.readback == 'on'
    cfg.setdefault('spark',{})['enabled'] = args.readback == 'on'
    if args.spark_cache is not None:
        cfg['spark']['cache'] = args.spark_cache
    if args.duration < 0 or args.trials < 0 or args.trial_seconds <= 0:
        raise ValueError("Duration and trial counts must be nonnegative, trial time positive")
    if args.camera_latency_ms is not None and args.camera_latency_ms < 0:
        raise ValueError("Camera latency cannot be negative")
    if args.headless and args.trials:
        raise ValueError("Interactive trials need the overlay window")
    if args.resume_trials and not args.trials:
        raise TrialResumeError('Use --trials with the original total when resuming.')
    if args.phrase and args.click:
        raise ValueError("Choose either a phrase or a clicked point")
    if args.trials and (args.phrase or args.click):
        raise ValueError("Trial mode chooses its own target")
    if args.trial_mode == 'voice' and args.trials and not cfg['voice']['enabled']:
        raise ValueError('Voice trials require microphone streaming')
    if args.speech_every < 0:
        raise ValueError('Speech interval cannot be negative')
    if args.record and args.mute:
        raise ValueError('Recording the output mix requires an active audio stream')
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
        if args.command == 'replay':
            from .replay import replay_session, ReplayError
            try:
                replayed=replay_session(args.run_dir,strip=args.strip)
            except ReplayError as exc:
                print(f'Cannot replay: {exc}',file=sys.stderr)
                return 2
            if replayed['status']=='no_inputs':
                print('no replay inputs')
                return 0
            if replayed['status']=='diverged':
                print(f"replay divergence: step {replayed['step']}, field {replayed['field']}; "
                      f"logged={json.dumps(replayed['logged'])[:300]}; replayed={json.dumps(replayed['replayed'])[:300]}")
                return 1
            print(f"replay OK: {replayed['steps']} steps, {replayed['effects']} effects, "
                  f"{replayed['state_changes']} state changes, read-backs {replayed['readbacks']}")
            if replayed.get('stripped'): print('Events-only copy: '+replayed['stripped'])
            return 0
        if args.command == "report":
            if args.html:
                from .session_report import write_html
                print(json.dumps(write_html(args.run_dir,args.html),indent=2))
                return 0
            from .session_report import summarize
            print(json.dumps(summarize(args.run_dir), indent=2))
            return 0
        if args.command == 'speak':
            return speech_probe(args)
        if args.command == 'readback-probe':
            from .spark_probe import probe
            return probe(args)
        cfg = configure(args)
        from .runner import Runner
        return Runner(cfg, args).run()
    except TrialResumeError as exc:
        print(f'Cannot resume trials: {exc}', file=sys.stderr)
        return 2
    except ImportError:
        print('Shared session modules are not available. Update to the integrated session build.', file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        # Device/config errors are actionable; never echo arbitrary network exception bodies.
        from .startup import explain, speak_once
        spoken, printed = explain(exc)
        print(f"Cannot start Reach ({type(exc).__name__}). {printed}", file=sys.stderr)
        if args.command == 'run' and not getattr(args, 'mute', False):
            speak_once(spoken)
        return 2


def speech_probe(args):
    from datetime import datetime, timezone
    from .audio import CueEngine
    from .tts import Speaker
    from .log import EventLog
    if args.count < 1 or args.count > 100:
        raise ValueError('Speech count must be between 1 and 100')
    cfg = load_config(args.config)
    root = args.log_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    audio = CueEngine(cfg)
    with EventLog(root, save_frames=False) as log:
        speaker = Speaker(cfg, audio, log)
        log.write_manifest({'mount': 'fixed', 'source': 'speech_only', 'objects': cfg.get('objects', []),
                            'prediction': 0, 'latency_note': 'Speech starts at the output callback; acoustic output delay is separate.'})
        try:
            audio.start()
            speaker.start()
            for _ in range(args.count):
                speaker.say(args.text, priority=1)
                while speaker.busy:
                    speaker.poll()
                    if audio.stats()['error']:
                        raise RuntimeError('Audio callback failed')
                    time.sleep(.005)
            speaker.poll()
            speech_stats = speaker.stats()
            result = {'completed': (not speech_stats['fallback'] and not speech_stats['error']
                                   and speech_stats['engine'] != 'null' and audio.stats()['callback_errors'] == 0), 'audio_measured': True,
                      'audio': audio.stats(), 'speech': speaker.stats(), 'trials': []}
            print(json.dumps(result, indent=2))
            (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        finally:
            speaker.stop()
            audio.stop()
    print(f'Speech run saved: {root}')
    return 0 if result['completed'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
