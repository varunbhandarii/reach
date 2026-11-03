"""Interactive device choice that writes the ignored local configuration."""

from pathlib import Path
import sys

import yaml

from .config import PROJECT_ROOT


def camera_candidates(indices=range(5), capture_factory=None):
    import cv2
    factory = capture_factory or cv2.VideoCapture
    backend = (cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else
               cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    found = []
    for index in indices:
        cap = factory(index, backend)
        try:
            ok, image = cap.read() if cap.isOpened() else (False, None)
            if ok and image is not None:
                found.append((index, f'{image.shape[1]}x{image.shape[0]}'))
        finally:
            cap.release()
    return found


def audio_candidates():
    import sounddevice as sd
    hosts, devices = sd.query_hostapis(), sd.query_devices()
    name = 'Windows WASAPI' if sys.platform == 'win32' else 'Core Audio' if sys.platform == 'darwin' else 'ALSA'
    host = next((i for i, h in enumerate(hosts) if h['name'] == name), None)
    outputs = [d['name'] for d in devices if d['hostapi'] == host and d['max_output_channels'] >= 2]
    inputs = [d['name'] for d in devices if d['hostapi'] == host and d['max_input_channels'] >= 1]
    return outputs, inputs


def _choose(prompt, options, input_fn):
    for i, label in enumerate(options, 1):
        print(f'  [{i}] {label}')
    answer = input_fn(prompt).strip()
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return int(answer)-1, None
    return None, answer or None


def run_setup(path=None, *, input_fn=input, cameras=None, audio=None):
    path = Path(path) if path else PROJECT_ROOT/'config.local.yaml'
    local = yaml.safe_load(path.read_text(encoding='utf-8')) if path.exists() else {}
    local = local if isinstance(local, dict) else {}
    cameras = camera_candidates() if cameras is None else cameras
    outputs, inputs = audio_candidates() if audio is None else audio
    print('Cameras:')
    index, typed = _choose('Camera number, an Android stream URL, or Enter to keep: ',
                           [f'camera {i} ({size})' for i, size in cameras], input_fn)
    camera = local.setdefault('camera', {})
    if index is not None:
        camera.update(source='iphone_continuity' if sys.platform == 'darwin' else 'usb_webcam',
                      device_index=cameras[index][0], backend='auto')
    elif typed and typed.startswith(('http://', 'https://')):
        camera.update(source='android_mjpeg', url=typed, backend='auto')
    print('Stereo outputs (wired headphones work best):')
    index, _ = _choose('Output number, or Enter to keep: ', outputs, input_fn)
    if index is not None:
        local.setdefault('audio', {})['output_device'] = outputs[index]
    print('Microphones (on a Mac choose the MacBook microphone, not AirPods):')
    index, _ = _choose('Microphone number, or Enter to keep: ', inputs, input_fn)
    if index is not None:
        if 'airpods' in inputs[index].casefold():
            print('AirPods microphones switch AirPods to call audio, which breaks stereo guidance; not saved.')
        else:
            local.setdefault('voice', {})['input_device'] = inputs[index]
    path.write_text(yaml.safe_dump(local, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'Saved {path.name}. Start with: uv run reach start')
    return path
