"""Render canned screen states to images for design review, without a camera or network."""

from pathlib import Path
import time

import cv2
import numpy as np

from .overlay import Captions, Screen
from .types import CueParams, Decision, HandObs, Instance, TargetObs


def synthetic_table(width=640, height=480):
    """A plain table with a wallet, two pens and a hand, so previews never show a real home."""
    image = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        image[y] = (40+y//12, 60+y//10, 85+y//9)
    cv2.rectangle(image, (300, 190), (390, 270), (40, 45, 60), -1)
    cv2.rectangle(image, (300, 190), (390, 270), (25, 28, 38), 3)
    cv2.line(image, (470, 150), (520, 300), (200, 160, 60), 9)
    cv2.line(image, (120, 120), (170, 220), (60, 60, 200), 9)
    cv2.ellipse(image, (180, 400), (70, 45), 20, 0, 360, (120, 150, 200), -1)
    cv2.ellipse(image, (230, 340), (16, 60), 30, 0, 360, (125, 155, 205), -1)
    return image


def scenarios():
    wallet = Instance('a', (300, 190, 390, 270), .92, None)
    pens = (Instance('b', (465, 145, 525, 305), .81, None), Instance('c', (115, 115, 175, 225), .79, None))
    hand = HandObs(1, True, (255, 285), (200, 380), (), (0, 0), 1)
    far = CueParams(.6, 520, 3, False, 0, .2)
    near = CueParams(.2, 460, 10, False, .12, .2)
    decide = lambda state, cue=None, d=None: Decision(0, 1, state, None, None, d, cue, None, None)
    voice = {'connected': True, 'stereo_ok': True, 'audio_route': {'status': 'ok'}}
    base = {'voice': voice, 'capture_fps': 29.4, 'hand_ms_p95': 21, 'motion_ms_p95': 9, 'images_sent': 42,
            'estimated_cost_usd': .118, 'reference_ready': True}
    target = lambda *instances, chosen=0, phrase='wallet': TargetObs(1, phrase, tuple(instances), chosen, 'sam', 0)
    return [
        ('01-ready', None, None, decide('IDLE'), {**base}, ('', "Reach is ready. Say find and an object, or what's on the table.")),
        ('02-searching', None, None, decide('SEARCHING'), {**base, 'phrase': 'wallet'}, ('find my wallet', 'Looking for the wallet.')),
        ('03-two-pens', target(*pens, phrase='pen'), hand, decide('DISAMBIGUATING'), {**base, 'phrase': 'pen'},
         ('find the pen', 'I see two pens. One on your left, one on your right. Say left or right.')),
        ('04-guiding-far', target(wallet), hand, decide('GUIDING', far, .4), {**base, 'phrase': 'wallet'},
         ('', 'Found the wallet, on your right, closer to you.')),
        ('05-guiding-near', target(wallet), HandObs(1, True, (330, 235), (300, 330), (), (0, 0), 1),
         decide('GUIDING', near, .05), {**base, 'phrase': 'wallet'}, ('', '')),
        ('06-hand-lost', target(wallet), None, decide('HAND_LOST'), {**base, 'phrase': 'wallet'}, ('', 'Bring your hand into view.')),
        ('07-stereo-lost', target(wallet), hand, decide('GUIDING', far, .4),
         {**base, 'phrase': 'wallet', 'route_ok': False, 'voice': {**voice, 'stereo_ok': False}},
         ('', 'Stereo lost. Check your headphones.')),
        ('08-confirming', target(wallet), HandObs(1, True, (340, 230), (300, 330), (), (0, 0), 1), decide('CONFIRMING'),
         {**base, 'phrase': 'wallet', 'readback': {'busy': True}}, ('', '')),
        ('09-matched', target(wallet), HandObs(1, True, (340, 230), (300, 330), (), (0, 0), 1), decide('DONE'),
         {**base, 'phrase': 'wallet', 'readback': {'last': {'matches': True, 'object_name': 'brown wallet', 'latency_ms': 1840}}},
         ('', 'You have the wallet.')),
        ('10-scene-list', None, None, decide('IDLE'),
         {**base, 'scene': {'last': {'objects': [('brown wallet', 'middle'), ('blue pen', 'right'), ('red pen', 'left')]}}},
         ("what's on the table", 'I can see a red pen on your left, a brown wallet in the middle and a blue pen on your right.')),
        ('11-hindi', target(wallet), hand, decide('GUIDING', far, .4), {**base, 'phrase': 'wallet'},
         ('मेरा बटुआ ढूंढो', 'बटुआ की जगह पता चल गई है: दाईं ओर, आपके पास।')),
    ]


def render(out_dir, background=None, scales=(1, 2), modes=('presentation', 'debug'), timing=300):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(background)) if background else None
    if image is None:
        image = synthetic_table()
    screen = Screen()
    written = []
    for scale in scales:
        for mode in modes:
            for name, target, hand, decision, stats, (heard, said) in scenarios():
                captions = Captions()
                captions.update(0, heard=heard or None, speaking=said or None, state=decision.state)
                low = name.endswith('guiding-far') and mode == 'presentation'
                for suffix, low_vision in (('', False), ('-low-vision', True)) if low else (('', False),):
                    canvas = screen.draw(image, hand, target, decision, stats, scale=scale, now=1/12,
                                         mode=mode, low_vision=low_vision, captions=captions)
                    path = out/f'{name}{suffix}-{mode}-x{scale}.png'
                    cv2.imwrite(str(path), canvas)
                    written.append(path)
    costs = []
    name, target, hand, decision, stats, (heard, said) = scenarios()[3]
    captions = Captions()
    captions.update(0, speaking=said, state=decision.state)
    for i in range(timing):
        start = time.perf_counter()
        screen.draw(image, hand, target, decision, stats, scale=2, now=i/30, mode='presentation', captions=captions)
        costs.append((time.perf_counter()-start)*1000)
    return written, (float(np.median(costs)), float(np.percentile(costs, 95))) if costs else (None, None)
