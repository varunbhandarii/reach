"""Two-pane guidance screen: the camera stays uncovered; status, captions and sound live in a side panel."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .controller import target_center
from .masks import decode_rle

FONT_DIR = Path(__file__).resolve().parent/'assets'/'fonts'

# Okabe-Ito colour-blind-safe palette, as RGB; every mark also has a shape or label.
BG = (16, 20, 24)
CARD = (28, 34, 41)
TEXT = (245, 247, 250)
MUTED = (170, 180, 190)
NEUTRAL = (120, 130, 140)
YELLOW = (240, 228, 66)
SKY = (86, 180, 233)
GREEN = (0, 158, 115)
ORANGE = (230, 159, 0)
VERMILLION = (213, 94, 0)
BLUE = (0, 114, 178)
PURPLE = (204, 121, 167)

KEYS = ('1-4 find  ·  D describe  ·  C cancel  ·  V talk', 'M mute  ·  U view  ·  L low vision  ·  Q quit')


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def sound_values(cue,now,enabled=True):
    if cue is None or not enabled:
        return {'active':False,'pan':0.,'pitch':.5,'pulse':False,'on_line':False}
    return {'active':True,'pan':max(-1.,min(1.,cue.pan)),
        'pitch':max(0.,min(1.,math.log(max(220.,min(880.,cue.freq_hz))/220)/math.log(4))),
        'pulse':bool(math.sin(2*math.pi*cue.pulse_hz*now)>0),'on_line':cue.noise>0}


def stereo_levels(pan):
    """Equal-power left and right levels, matching the audio engine's pan law."""
    theta = (max(-1., min(1., pan))+1)*math.pi/4
    return math.cos(theta), math.sin(theta)


def pitch_color(pitch):
    """Distinct colours rather than a blend: farther is sky blue, closer is orange."""
    return SKY if pitch >= .55 else ORANGE if pitch <= .45 else TEXT


def pulse_word(hz):
    return 'slow' if hz < 4 else 'medium' if hz < 8 else 'fast'


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    pane_width: int
    image: tuple  # x, y, w, h of the camera image inside the pane
    factor: float  # source pixels to canvas pixels
    unit: float  # 1.0 at 1080 rows

    @property
    def panel(self):
        return self.pane_width, 0, self.width-self.pane_width, self.height

    def to_canvas(self, xy):
        return (round(self.image[0]+xy[0]*self.factor), round(self.image[1]+xy[1]*self.factor))

    def to_source(self, xy):
        x, y = (xy[0]-self.image[0])/self.factor, (xy[1]-self.image[1])/self.factor
        width, height = self.image[2]/self.factor, self.image[3]/self.factor
        return (x, y) if 0 <= x < width and 0 <= y < height else None


def layout_for(source_width, source_height, scale=1):
    if scale not in (1, 2):
        raise ValueError('Overlay scale must be one or two')
    width, height = (1920, 1080) if scale == 2 else (1280, 720)
    pane = height*4//3
    fit = min(pane/source_width, height/source_height)
    w, h = round(source_width*fit), round(source_height*fit)
    return Layout(width, height, pane, ((pane-w)//2, (height-h)//2, w, h), fit, height/1080)


class TextRenderer:
    """Pillow text with bundled fonts; falls back to OpenCV text when unavailable."""

    def __init__(self, font_dir=FONT_DIR):
        self._fonts, self._lengths, self._masks, self._wraps = {}, {}, {}, {}
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._Image, self._Draw, self._Font = Image, ImageDraw, ImageFont
            self._paths = {'regular': Path(font_dir)/'AtkinsonHyperlegible-Regular.ttf',
                           'bold': Path(font_dir)/'AtkinsonHyperlegible-Bold.ttf',
                           'devanagari': Path(font_dir)/'NotoSansDevanagari-Regular.ttf'}
            self.available = self._paths['regular'].is_file() and self._paths['bold'].is_file()
        except ImportError:
            self.available = False

    def font(self, px, bold=False, text=''):
        role = ('devanagari' if any('ऀ' <= c <= 'ॿ' for c in text)
                and self._paths['devanagari'].is_file() else 'bold' if bold else 'regular')
        key = role, int(px)
        if key not in self._fonts:
            # Complex-script shaping is only needed for Devanagari; basic layout is much faster.
            engine = self._Font.Layout.RAQM if role == 'devanagari' else self._Font.Layout.BASIC
            self._fonts[key] = self._Font.truetype(str(self._paths[role]), int(px), layout_engine=engine)
        return self._fonts[key]

    def width(self, text, px, bold=False):
        if not self.available:
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, px/30, max(1, round(px/15)))[0][0]
        key = text, int(px), bold
        if key not in self._lengths:
            if len(self._lengths) > 4000:
                self._lengths.clear()
            self._lengths[key] = self.font(px, bold, text).getlength(text)
        return self._lengths[key]

    def mask(self, text, px, bold=False):
        """Rendered text shape, cached; colour is applied when it is pasted."""
        key = text, int(px), bold
        if key not in self._masks:
            if len(self._masks) > 2000:
                self._masks.clear()
            font = self.font(px, bold, text)
            left, top, right, bottom = font.getbbox(text)
            shape = self._Image.new('L', (max(1, right-left), max(1, bottom-top)), 0)
            self._Draw.Draw(shape).text((-left, -top), text, font=font, fill=255)
            self._masks[key] = shape, (left, top)
        return self._masks[key]

    def wrap(self, text, px, limit, bold=False, lines=3):
        key = str(text), int(px), int(limit), bold, lines
        if key not in self._wraps:
            if len(self._wraps) > 2000:
                self._wraps.clear()
            self._wraps[key] = self._wrap(*key)
        return self._wraps[key]

    def _wrap(self, text, px, limit, bold=False, lines=3):
        out = []
        for word in str(text).split(' '):
            if not word:
                if out:
                    out[-1] += ' '
                continue
            if out and self.width(out[-1]+' '+word, px, bold) <= limit:
                out[-1] += ' '+word
            else:
                out.append(word)
        if len(out) > lines:
            out = out[:lines]
            out[-1] = out[-1].rstrip('.,')+'...'
        return out


def fade(color, alpha, base=CARD):
    return tuple(round(b+(c-b)*alpha) for c, b in zip(color, base))


class Captions:
    """What the user said and what Reach said; never shown after it stops being current."""

    HEARD_S, SAID_AFTER_S, FADE_S = 4.0, 2.0, .5

    def __init__(self):
        self.heard = self.heard_t = self._heard_source = None
        self.said = self.said_start = self.said_end = None
        self.state = None

    def update(self, now, *, heard=None, speaking=None, state=None):
        if heard and heard != self._heard_source:
            self.heard, self.heard_t, self._heard_source = heard, now, heard
        if speaking:
            if speaking != self.said or self.said_end is not None:
                self.said, self.said_start, self.said_end = speaking, now, None
        elif self.said and self.said_end is None:
            self.said_end = now
        if state != self.state:
            if self.state is not None and self.said and self.said_end is not None:
                self.said = None  # A finished phrase from the previous state is outdated.
            self.state = state

    def _alpha(self, now, until):
        # Three fade steps keep the cached panel from being redrawn on every frame.
        left = until-now
        return 0. if left <= 0 else 1. if left >= self.FADE_S else .6 if left >= self.FADE_S/2 else .3

    def visible(self, now):
        heard = said = None
        if self.heard and self._alpha(now, self.heard_t+self.HEARD_S) > 0:
            heard = self.heard, self._alpha(now, self.heard_t+self.HEARD_S)
        if self.said:
            alpha = 1. if self.said_end is None else self._alpha(now, self.said_end+self.SAID_AFTER_S)
            if alpha > 0:
                said = self.said, alpha
        return {'heard': heard, 'said': said}


def state_card(decision, target, stats, muted=False):
    voice = stats.get('voice', {})
    phrase = stats.get('phrase') or (target.phrase if target else '') or ''
    name = 'the point' if phrase in ('', 'clicked point') else f'the {phrase}'
    if stats.get('error'):
        return str(stats['error'])[:80], VERMILLION
    if voice.get('stereo_ok') is False or stats.get('route_ok') is False:
        return 'Stereo lost: check headphones', VERMILLION
    state = decision.state if decision else 'IDLE'
    last = (stats.get('readback') or {}).get('last') or {}
    labels = {'IDLE': ('Ready', NEUTRAL), 'SEARCHING': (f'Looking for {name}', SKY),
              'DISAMBIGUATING': (f'Which {phrase or "one"}? Say left or right', YELLOW),
              'GUIDING': (f'Guiding to {name}', GREEN), 'HAND_LOST': ('Hand out of view', ORANGE),
              'TARGET_LOST': (f'Lost {name}, hold still', ORANGE),
              'CONFIRMING': ('Touching, checking...', YELLOW), 'CONTACT': ('Touching, checking...', YELLOW),
              'DESCRIBING': ('Looking around...', SKY)}
    if state == 'DONE':
        text, color = ((f'Confirmed: {phrase}', GREEN) if last.get('matches') is True else
                       (f'Not {name}', VERMILLION) if last.get('matches') is False else (f'Touching {name}', GREEN))
    else:
        text, color = labels.get(state, (state.replace('_', ' ').title(), NEUTRAL))
    return (text+' (muted)' if muted else text), color


def check_lines(decision, target, stats, trial):
    if trial:
        return [(line, TEXT) for line in trial[:3]]
    state = decision.state if decision else 'IDLE'
    readback = stats.get('readback') or {}
    if readback.get('busy'):
        return [('Checking the touched object...', YELLOW)]
    last = readback.get('last')
    if last and state in ('CONFIRMING', 'DONE', 'CONTACT'):
        name = last.get('object_name') or 'object'
        latency = f"  {last['latency_ms']/1000:.1f} s" if last.get('latency_ms') else ''
        if last.get('error'):
            return [(f'Check unavailable ({last["error"]})', MUTED)]
        if last.get('matches') is True:
            return [(f'Matched: {name}{latency}', GREEN)]
        if last.get('matches') is False:
            return [(f'Not it: looks like {name}{latency}', VERMILLION)]
        return [(f'Unsure: {name}{latency}', YELLOW)]
    if state == 'DISAMBIGUATING' and target and len(target.instances) > 1:
        return [(f'{len(target.instances)} found: say left or right', YELLOW)]
    objects = ((stats.get('scene') or {}).get('last') or {}).get('objects') or ()
    if objects and state in ('IDLE', 'DESCRIBING', 'SEARCHING'):
        return [(f'{i}. {name}, {position}', TEXT) for i, (name, position) in enumerate(objects[:6], 1)]
    return []


def chips(stats, camera_ok=True):
    voice = stats.get('voice', {})
    online = None if voice.get('enabled') is False else bool(voice.get('connected'))
    route = (voice.get('audio_route') or {}).get('status') if isinstance(voice.get('audio_route'), dict) else voice.get('audio_route')
    stereo = False if voice.get('stereo_ok') is False else None if route == 'unknown' else True
    return [('Voice', online, 'on' if online else 'off' if online is None else 'connecting'),
            ('Stereo', stereo, 'ok' if stereo else 'unknown' if stereo is None else 'lost'),
            ('Camera', camera_ok, 'ok' if camera_ok else 'no frames')]


def debug_lines(stats):
    def number(key, digits=0):
        value = stats.get(key)
        return f'{value:.{digits}f}' if isinstance(value, (int, float)) else '-'
    ref = stats.get('reference_ready')
    return [f"{number('capture_fps')} fps  ·  hand p95 {number('hand_ms_p95')} ms  ·  motion p95 {number('motion_ms_p95')} ms",
            f"images {stats.get('images_sent', 0)}  ·  cost ~${stats.get('estimated_cost_usd', 0):.3f}  ·  "
            f"ref {'ready' if ref else 'waiting' if ref is False else 'off'}  ·  volume {stats.get('master_db', 0):+.0f} dB",
            *KEYS]


class Screen:
    """Draws one frame; caches the text panel until its content changes."""

    def __init__(self, renderer=None):
        self.text = renderer or TextRenderer()
        self._panel_key = self._panel = self._sound_box = None

    def _panel_image(self, layout, content):
        key = (layout.width, layout.height, repr(content))
        if key == self._panel_key:
            return self._panel, self._sound_box
        x0, _, width, height = layout.panel
        u = layout.unit
        pad, gap = round(24*u), round(16*u)
        inner = width-2*pad
        sizes = {'brand': 24*u, 'state': 40*u, 'label': 20*u, 'body': 30*u, 'small': 22*u, 'debug': 18*u}
        items = []  # (kind, x, y, w, h, payload)
        y = pad
        items.append(('text', pad, y, 'Reach', sizes['brand'], MUTED, True))
        y += round(40*u)
        state_text, state_color = content['state']
        lines = self.text.wrap(state_text, sizes['state'], inner-round(18*u), bold=True, lines=2)
        card_h = round(len(lines)*sizes['state']*1.25+28*u)
        items.append(('card', pad, y, inner, card_h, CARD))
        items.append(('bar', pad, y, round(8*u), card_h, state_color))
        for i, line in enumerate(lines):
            items.append(('text', pad+round(22*u), y+round(14*u)+round(i*sizes['state']*1.25), line,
                          sizes['state'], TEXT, True))
        y += card_h+gap
        for label, entry in (('YOU SAID', content['heard']), ('REACH', content['said'])):
            if entry is None:
                continue
            words, alpha = entry
            items.append(('text', pad, y, label, sizes['label'], fade(MUTED, alpha, BG), True))
            y += round(28*u)
            for line in self.text.wrap(words, sizes['body'], inner, lines=3):
                items.append(('text', pad, y, line, sizes['body'], fade(TEXT, alpha, BG), False))
                y += round(sizes['body']*1.25)
            y += gap
        items.append(('text', pad, y, 'SOUND', sizes['label'], MUTED, True))
        y += round(28*u)
        sound_h = round(150*u)
        sound_box = (x0+pad, y, inner, sound_h)
        items.append(('card', pad, y, inner, sound_h, CARD))
        sound = content['sound']
        items.append(('text', pad+round(14*u), y+round(10*u), 'L', sizes['small'], MUTED, True))
        items.append(('text', pad+round(84*u), y+round(10*u), 'R', sizes['small'], MUTED, True))
        ladder_x = pad+round(170*u)
        items.append(('text', ladder_x+round(18*u), y+round(8*u), 'farther', sizes['small'], MUTED, False))
        items.append(('text', ladder_x+round(18*u), y+sound_h-round(36*u), 'closer', sizes['small'], MUTED, False))
        status = sound['status']
        items.append(('text', pad+inner-round(120*u), y+sound_h-round(36*u), status, sizes['small'],
                      VERMILLION if status in ('stereo lost', 'muted') else TEXT, False))
        if sound['on_line']:
            items.append(('text', pad+inner-round(120*u), y+round(48*u), 'on line', sizes['small'], GREEN, True))
        y += sound_h+gap
        checks = content['check']
        if checks or content.get('inset'):
            items.append(('text', pad, y, 'CHECK' if checks else 'ZOOM', sizes['label'], MUTED, True))
            y += round(28*u)
            for line, color in checks:
                for part in self.text.wrap(line, sizes['small'], inner, lines=2):
                    items.append(('text', pad, y, part, sizes['small'], color, False))
                    y += round(sizes['small']*1.3)
            y += gap
        inset_y = y
        bottom = height-pad
        if content['debug']:
            debug_rows = []
            for line in content['debug']:
                debug_rows.extend(self.text.wrap(line, sizes['debug'], inner, lines=2))
            bottom -= round(len(debug_rows)*sizes['debug']*1.35)+gap
            for i, line in enumerate(debug_rows):
                items.append(('text', pad, bottom+gap+round(i*sizes['debug']*1.35), line, sizes['debug'], MUTED, False))
        chip_h = round(40*u)
        chip_w = (inner-2*round(8*u))//3
        for i, (name, ok, word) in enumerate(content['chips']):
            cx = pad+i*(chip_w+round(8*u))
            color = GREEN if ok else NEUTRAL if ok is None else VERMILLION
            items.append(('card', cx, bottom-chip_h, chip_w, chip_h, CARD))
            items.append(('dot', cx+round(16*u), bottom-chip_h//2, round(6*u), 0, color))
            items.append(('text', cx+round(30*u), bottom-chip_h+round(8*u), f'{name} {word}', sizes['debug'], TEXT, False))
        panel = self._render(width, height, items)
        self._panel_key, self._panel, self._sound_box = key, panel, sound_box
        self._inset_box = (x0+pad, inset_y, inner, max(0, bottom-chip_h-gap-inset_y))
        return panel, sound_box

    def _render(self, width, height, items):
        if self.text.available:
            image = self.text._Image.new('RGB', (width, height), BG)
            draw = self.text._Draw.Draw(image)
            for kind, x, y, *rest in items:
                if kind == 'card':
                    w, h, color = rest
                    draw.rounded_rectangle((x, y, x+w, y+h), radius=max(2, h//8), fill=color)
                elif kind == 'bar':
                    w, h, color = rest
                    draw.rectangle((x, y, x+w, y+h), fill=color)
                elif kind == 'dot':
                    r, _, color = rest
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
                else:
                    text, px, color, bold = rest
                    shape, (ox, oy) = self.text.mask(text, px, bold)
                    image.paste(color, (int(x+ox), int(y+oy)), shape)
            return np.asarray(image)[:, :, ::-1].copy()
        panel = np.zeros((height, width, 3), np.uint8)
        panel[:] = bgr(BG)
        for kind, x, y, *rest in items:
            if kind in ('card', 'bar'):
                w, h, color = rest
                cv2.rectangle(panel, (x, y), (x+w, y+h), bgr(color), -1)
            elif kind == 'dot':
                r, _, color = rest
                cv2.circle(panel, (x, y), r, bgr(color), -1)
            else:
                text, px, color, bold = rest
                cv2.putText(panel, text, (x, y+round(px*.85)), cv2.FONT_HERSHEY_SIMPLEX, px/34, bgr(color),
                            max(1, round(px/(14 if bold else 20))), cv2.LINE_AA)
        return panel

    def draw(self, image, hand, target, decision, stats, *, muted=False, trial=None, stale_box=None,
             scale=1, now=None, mode='debug', low_vision=False, captions=None):
        now = time.perf_counter() if now is None else now
        layout = layout_for(image.shape[1], image.shape[0], scale)
        u = layout.unit
        canvas = np.empty((layout.height, layout.width, 3), np.uint8)
        x, y, w, h = layout.image
        if (w, h) != (layout.pane_width, layout.height):
            if getattr(self, '_pane_bg', None) is None or self._pane_bg.shape[:2] != (layout.height, layout.pane_width):
                self._pane_bg = np.empty((layout.height, layout.pane_width, 3), np.uint8)
                self._pane_bg[:] = bgr(BG)
            canvas[:, :layout.pane_width] = self._pane_bg
        view = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if low_vision:
            view = cv2.convertScaleAbs(view, alpha=.6)
        canvas[y:y+h, x:x+w] = view
        voice = stats.get('voice', {})
        enabled = not muted and stats.get('route_ok', True) and voice.get('stereo_ok', True)
        values = sound_values(decision.cue if decision else None, now, enabled)
        self._marks(canvas, layout, hand, target, decision, stats, stale_box, values, low_vision)
        visible = captions.visible(now) if captions else {'heard': None, 'said': None}
        if captions is None:
            caption = (stats.get('speech') or {}).get('text')
            visible['said'] = (caption, 1.) if caption else None
        cue = decision.cue if decision else None
        status = ('muted' if muted else 'stereo lost' if not enabled else
                  pulse_word(cue.pulse_hz) if values['active'] else 'silent')
        content = {'state': state_card(decision, target, stats, muted), 'heard': visible['heard'],
                   'said': visible['said'], 'sound': {'status': status, 'on_line': values['on_line']},
                   'check': check_lines(decision, target, stats, trial),
                   'chips': chips(stats, camera_ok=image is not None),
                   'debug': debug_lines(stats) if mode == 'debug' else [],
                   'inset': low_vision and target_center(target) is not None}
        panel, sound_box = self._panel_image(layout, content)
        canvas[:, layout.pane_width:] = panel
        self._sound(canvas, sound_box, values, u)
        if low_vision:
            self._low_vision(canvas, layout, image, target, content['state'])
        return canvas

    def _marks(self, canvas, layout, hand, target, decision, stats, stale_box, values, low_vision):
        u = layout.unit
        thick = lambda n: max(1, round((n*2.5 if low_vision else n)*u))
        center = target_center(target)
        if target:
            for index, instance in enumerate(target.instances):
                p0, p1 = layout.to_canvas(instance.box_xyxy[:2]), layout.to_canvas(instance.box_xyxy[2:])
                if index == target.chosen:
                    self._tint(canvas, instance, p0, p1)
                    cv2.rectangle(canvas, p0, p1, bgr(YELLOW), thick(4))
                else:
                    cv2.rectangle(canvas, p0, p1, bgr(BLUE), thick(2))
                    badge = (p0[0]+round(16*u), p0[1]+round(16*u))
                    cv2.circle(canvas, badge, round(15*u), bgr(BLUE), -1)
                    cv2.putText(canvas, str(index+1), (badge[0]-round(7*u), badge[1]+round(8*u)),
                                cv2.FONT_HERSHEY_SIMPLEX, .7*u*1.5, bgr(TEXT), max(1, round(2*u)), cv2.LINE_AA)
        elif stale_box:
            cv2.rectangle(canvas, layout.to_canvas(stale_box[:2]), layout.to_canvas(stale_box[2:]), bgr(NEUTRAL), thick(2))
        for item in stats.get('excluded_instances', []):
            x0, y0, x1, y1 = item['box_xyxy']
            cv2.line(canvas, layout.to_canvas((x0, y0)), layout.to_canvas((x1, y1)), bgr(PURPLE), thick(3))
            cv2.line(canvas, layout.to_canvas((x0, y1)), layout.to_canvas((x1, y0)), bgr(PURPLE), thick(3))
        if hand and hand.present and hand.fingertip_px:
            tip = layout.to_canvas(hand.fingertip_px)
            if center:
                cv2.arrowedLine(canvas, tip, layout.to_canvas(center), bgr(SKY), thick(4), cv2.LINE_AA, tipLength=.06)
            if values['active']:
                closeness = 1-min(1., (decision.d if decision and decision.d is not None else .5)/.5)
                color = pitch_color(values['pitch'])
                if values['pulse']:
                    cv2.circle(canvas, tip, round((24+16*closeness)*u), bgr(color), thick(4), cv2.LINE_AA)
                if values['on_line']:
                    for offset in range(-160, 161, 32):
                        a = (tip[0]+round(offset*u), tip[1])
                        cv2.line(canvas, a, (a[0]+round(16*u), a[1]), bgr(GREEN), thick(2))
            cv2.circle(canvas, tip, round(9*u), bgr(TEXT), -1)
            cv2.circle(canvas, tip, round(7*u), bgr(ORANGE), -1)

    @staticmethod
    def _tint(canvas, instance, p0, p1):
        if instance.mask_rle is None:
            return
        try:
            mask = decode_rle(instance.mask_rle)
        except Exception:
            return
        w, h = p1[0]-p0[0], p1[1]-p0[1]
        if w <= 0 or h <= 0 or p0[0] < 0 or p0[1] < 0 or p1[0] > canvas.shape[1] or p1[1] > canvas.shape[0]:
            return
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        region = canvas[p0[1]:p1[1], p0[0]:p1[0]]
        region[mask] = (region[mask]*.75+np.array(bgr(YELLOW))*.25).astype(np.uint8)

    @staticmethod
    def _sound(canvas, box, values, u):
        x, y, w, h = box
        top, bottom = y+round(44*u), y+h-round(16*u)
        left, right = stereo_levels(values['pan']) if values['active'] else (0., 0.)
        for i, level in enumerate((left, right)):
            bx = x+round((14+70*i)*u)
            cv2.rectangle(canvas, (bx, top), (bx+round(28*u), bottom), bgr(BG), -1)
            fill = round((bottom-top)*level)
            if fill:
                cv2.rectangle(canvas, (bx, bottom-fill), (bx+round(28*u), bottom), bgr(SKY), -1)
        lx = x+round(170*u)
        cv2.line(canvas, (lx, y+round(20*u)), (lx, y+h-round(20*u)), bgr(MUTED), max(1, round(3*u)))
        if values['active']:
            py = round(y+h-round(20*u)-(h-round(40*u))*values['pitch'])
            cv2.circle(canvas, (lx, py), round(10*u), bgr(pitch_color(values['pitch'])), -1)
        dot = (x+w-round(40*u), y+round(30*u))
        cv2.circle(canvas, dot, round(14*u), bgr(GREEN if values['pulse'] else NEUTRAL), -1)

    def _low_vision(self, canvas, layout, image, target, state):
        u = layout.unit
        band = round(110*u)
        canvas[:band, :layout.pane_width] = (canvas[:band, :layout.pane_width]*.25).astype(np.uint8)
        text, color = state
        if self.text.available:
            px = round(72*u)
            patch = self.text._Image.new('RGB', (layout.pane_width, band), (0, 0, 0))
            draw = self.text._Draw.Draw(patch)
            draw.text((round(24*u), round(14*u)), text, font=self.text.font(px, True, text), fill=YELLOW)
            array = np.asarray(patch)[:, :, ::-1]
            lit = array.max(axis=2) > 0
            canvas[:band, :layout.pane_width][lit] = array[lit]
        else:
            cv2.putText(canvas, text, (round(24*u), round(80*u)), cv2.FONT_HERSHEY_SIMPLEX, 2*u,
                        bgr(YELLOW), max(2, round(5*u)), cv2.LINE_AA)
        center = target_center(target)
        box = getattr(self, '_inset_box', None)
        if center is None or box is None or box[3] < 40:
            return
        x0, y0, x1, y1 = target.instances[target.chosen].box_xyxy
        mx, my = (x1-x0)*.3, (y1-y0)*.3
        crop = image[max(0, int(y0-my)):int(y1+my), max(0, int(x0-mx)):int(x1+mx)]
        if crop.size == 0:
            return
        bx, by, bw, bh = box
        fit = min(bw/crop.shape[1], bh/crop.shape[0], 2*layout.factor)
        zoom = cv2.resize(crop, (max(1, round(crop.shape[1]*fit)), max(1, round(crop.shape[0]*fit))))
        canvas[by:by+zoom.shape[0], bx:bx+zoom.shape[1]] = zoom
        cv2.rectangle(canvas, (bx, by), (bx+zoom.shape[1], by+zoom.shape[0]), bgr(YELLOW), max(2, round(4*u)))


_SCREEN = None


def draw_overlay(image, hand, target, decision, stats, *, muted=False, trial=None, stale_box=None,
                 scale=1, now=None, mode='debug', low_vision=False, captions=None):
    global _SCREEN
    if _SCREEN is None:
        _SCREEN = Screen()
    return _SCREEN.draw(image, hand, target, decision, stats, muted=muted, trial=trial, stale_box=stale_box,
                        scale=scale, now=now, mode=mode, low_vision=low_vision, captions=captions)
