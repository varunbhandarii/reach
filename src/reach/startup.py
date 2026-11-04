"""Plain startup problem messages, spoken when an output is available."""

import os
import sys

# (message fragments, spoken line or None, printed fix)
_RULES = (
    (('hand model missing',), 'The hand model is missing.',
     'Hand model missing: save hand_landmarker.task as models/hand_landmarker.task (see README), or set hand.model_path.'),
    (('camera stopped delivering',), 'The camera stream stopped.',
     'Camera stream stopped: keep the phone screen on and its camera app running, stay near the Wi-Fi router, '
     'then start again. A USB link is steadier: adb forward tcp:8080 tcp:8080 and camera.url http://127.0.0.1:8080/video.'),
    (('cannot open camera', 'no usable frames', 'camera.path', 'camera source',
      'pin orientation'), "I can't open the camera.",
     'Camera: check the source and permissions. Run `reach setup` to list cameras, or pass --source.'),
    (('model api key missing',), 'The API key is missing.',
     'API key: copy .env.example to .env and add your own Meta Model API key.'),
    (('input name', 'microphone', 'airpods'), 'Microphone unavailable. Use the keys.',
     'Microphone: set voice.input_device in config.local.yaml (the MacBook microphone on a Mac), or run with --no-voice.'),
    (('output name', 'no output device', 'stereo output', 'audio host api'), None,
     'Audio output: run `reach devices`, then set audio.output_device in config.local.yaml.'),
)


def explain(error):
    """Return (spoken line or None, printed fix) for a startup failure."""
    text = str(error).casefold()
    for fragments, spoken, printed in _RULES:
        if any(fragment in text for fragment in fragments):
            return spoken, printed
    return None, "Check the camera, hand model and audio device; details are in the run's events.jsonl."


def speak_once(text, cfg=None):
    """Best-effort spoken message through the default output, for failures before audio starts."""
    if not text or os.environ.get('PYTEST_CURRENT_TEST'):
        return False
    try:
        import sounddevice as sd
        if sys.platform == 'win32':
            from .tts import SapiEngine
            engine = SapiEngine((cfg or {}).get('tts', {}))
        elif sys.platform == 'darwin':
            from .say_engine import SayEngine
            engine = SayEngine((cfg or {}).get('tts', {}))
        else:
            return False
        samples = engine.synthesize(text, 48000)
        if hasattr(engine, 'close'):
            engine.close()
        sd.play(samples, 48000, blocking=True)
        return True
    except Exception:
        return False
