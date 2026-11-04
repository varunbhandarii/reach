"""Draw the system diagram once: an SVG for the README and a PNG for slides.

uv run python docs/architecture.py            # writes docs/architecture.svg
uv run python docs/architecture.py --png out.png --dark
"""

import argparse
from pathlib import Path

FONTS = Path(__file__).resolve().parents[1]/'src'/'reach'/'assets'/'fonts'
WIDTH, HEIGHT = 1600, 900

# Okabe-Ito colours: local work, Meta Model API, devices.
LOCAL, META, DEVICE = '#0072B2', '#D55E00', '#009E73'
THEMES = {'light': {'bg': '#FFFFFF', 'panel': '#F2F4F7', 'box': '#FFFFFF', 'text': '#101418', 'muted': '#4A5562', 'line': '#5B6573'},
          'dark': {'bg': '#101418', 'panel': '#1C2229', 'box': '#232A33', 'text': '#F5F7FA', 'muted': '#AAB4BE', 'line': '#AAB4BE'}}

GROUPS = [  # x, y, w, h, title, colour
    (30, 70, 250, 800, 'Devices', DEVICE),
    (310, 70, 860, 800, 'Laptop (Windows or macOS), all local', LOCAL),
    (1200, 70, 370, 800, 'Meta Model API', META),
]
BOXES = {  # id: x, y, w, h, title, detail lines, colour
    'camera': (50, 130, 210, 120, 'Phone camera', ['held in one hand', '640×480, about 29 fps', '~125 ms to the laptop'], DEVICE),
    'mic': (50, 380, 210, 100, 'Microphone', ['laptop mic', 'streamed at 24 kHz'], DEVICE),
    'ears': (50, 600, 210, 110, 'Headphones', ['stereo guide sound', 'earcons and speech'], DEVICE),
    'capture': (330, 130, 250, 120, 'Capture', ['newest-frame ring', 'reconnects a stalled', 'phone stream'], LOCAL),
    'hand': (610, 130, 250, 120, 'Hand tracking', ['MediaPipe landmarks', 'One Euro fingertip', 'p95 12 to 23 ms'], LOCAL),
    'motion': (890, 130, 260, 120, 'Target propagation', ['table homography,', 'hand masked out', 'p50 5 to 13 ms'], LOCAL),
    'voice': (330, 370, 250, 120, 'Voice client', ['endpointing stream', 'local grammar', 'speech end to command ~0.5 s'], LOCAL),
    'brain': (610, 360, 300, 140, 'Session state machine', ['deterministic, per hand frame', 'contact rule: inside, still,', 'mask overlap', 'every step logged'], LOCAL),
    'replay': (940, 370, 210, 120, 'Event log', ['exact replay of', 'every session'], LOCAL),
    'sound': (330, 590, 250, 130, 'Sound engine', ['pan: left or right', 'pitch: closer or farther', 'pulse: distance', '48 kHz, 5 ms blocks'], LOCAL),
    'speech': (610, 600, 250, 110, 'Local speech', ['Windows SAPI or', 'macOS say, cached', 'starts in ~5 ms'], LOCAL),
    'screen': (890, 600, 260, 110, 'Screen', ['camera view, captions,', 'sound and status'], LOCAL),
    'sam': (1220, 130, 330, 140, 'SAM 3.1', ['object from a noun phrase:', 'boxes, masks, confidence', 'p50 ~0.8 s, applied late', 'without losing the hand'], META),
    'transcribe': (1220, 360, 330, 140, 'Muse Voice Transcribe', ['realtime endpointing', 'keyword biasing', 'natural requests via Spark'], META),
    'spark': (1220, 590, 330, 160, 'Muse Spark 1.3', ['two-image touch check', '(before reach vs touch)', 'what is on the table', 'structured JSON, store off'], META),
}
# from, to, label, two-way, waypoints (a route around other boxes)
ARROWS = [
    ('camera', 'capture', 'frames', False, ()), ('capture', 'hand', '', False, ()), ('hand', 'motion', '', False, ()),
    ('motion', 'sam', '', True, ()),
    ('mic', 'voice', 'audio', False, ()),
    ('voice', 'transcribe', 'speech out, text back', True, ((455, 545), (1185, 545), (1185, 430)), (1035, 537)),
    ('hand', 'brain', 'fingertip', False, ()), ('motion', 'brain', 'target', False, ()), ('voice', 'brain', '', False, ()),
    ('brain', 'replay', '', False, ()), ('brain', 'spark', 'touch check', True, (), (1185, 580)),
    ('brain', 'sound', 'cue', False, ()), ('brain', 'speech', 'phrase', False, ()), ('speech', 'sound', '', False, ()),
    ('sound', 'ears', '', False, ()), ('brain', 'screen', '', False, ()),
]


def center(box):
    x, y, w, h = box[:4]
    return x+w/2, y+h/2


def edge(box, toward):
    """Point where the line from the box centre toward a point leaves the box."""
    x, y, w, h = box[:4]
    cx, cy = center(box)
    dx, dy = toward[0]-cx, toward[1]-cy
    if dx == dy == 0:
        return cx, cy
    scale = min((w/2)/abs(dx) if dx else float('inf'), (h/2)/abs(dy) if dy else float('inf'))
    return cx+dx*scale, cy+dy*scale


def segments():
    """Polylines with label positions: (points, label, two_way, label_at)."""
    out = []
    for a, b, label, both, via, *place in ARROWS:
        pa, pb = center(BOXES[a]), center(BOXES[b])
        first = via[0] if via else pb
        last = via[-1] if via else pa
        points = [edge(BOXES[a], first), *via, edge(BOXES[b], last)]
        longest = max(range(len(points)-1), key=lambda i: abs(points[i+1][0]-points[i][0])+abs(points[i+1][1]-points[i][1]))
        (x0, y0), (x1, y1) = points[longest], points[longest+1]
        out.append((points, label, both, place[0] if place else ((x0+x1)/2, (y0+y1)/2-8)))
    return out


def svg(theme='light'):
    t = THEMES[theme]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
             f'font-family="Atkinson Hyperlegible, Segoe UI, Arial, sans-serif">',
             '<title>Reach architecture</title>',
             '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
             f'<path d="M0 0L10 5L0 10z" fill="{t["line"]}"/></marker></defs>',
             f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["bg"]}"/>',
             f'<text x="30" y="45" font-size="30" font-weight="700" fill="{t["text"]}">Reach: from a spoken request to a guided touch</text>']
    for x, y, w, h, title, colour in GROUPS:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{t["panel"]}" stroke="{colour}" stroke-width="2"/>')
        parts.append(f'<text x="{x+16}" y="{y+32}" font-size="20" font-weight="700" fill="{colour}">{title}</text>')
    for points, label, both, (lx, ly) in segments():
        path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
        start = ' marker-start="url(#a)"' if both else ''
        parts.append(f'<polyline points="{path}" fill="none" stroke="{t["line"]}" stroke-width="2"{start} marker-end="url(#a)"/>')
        if label:
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" text-anchor="middle" fill="{t["muted"]}" '
                         f'paint-order="stroke" stroke="{t["panel"]}" stroke-width="5">{label}</text>')
    for x, y, w, h, title, lines, colour in BOXES.values():
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["box"]}" stroke="{colour}" stroke-width="2.5"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="8" height="{h}" rx="3" fill="{colour}"/>')
        parts.append(f'<text x="{x+20}" y="{y+30}" font-size="20" font-weight="700" fill="{t["text"]}">{title}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x+20}" y="{y+56+i*22}" font-size="16" fill="{t["muted"]}">{line}</text>')
    parts.append(f'<text x="30" y="{HEIGHT-8}" font-size="15" fill="{t["muted"]}">Times are measured medians or 95th percentiles '
                 'from our runs; the camera and SAM times dominate. Orange: Meta Model API. Blue: local. Green: devices.</text>')
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


def png(path, theme='dark', width=1920):
    from PIL import Image, ImageDraw, ImageFont
    t = THEMES[theme]
    s = width/WIDTH
    image = Image.new('RGB', (width, round(HEIGHT*s)), t['bg'])
    draw = ImageDraw.Draw(image)
    font = lambda px, bold=False: ImageFont.truetype(str(FONTS/('AtkinsonHyperlegible-Bold.ttf' if bold else 'AtkinsonHyperlegible-Regular.ttf')), round(px*s))
    S = lambda *v: tuple(round(a*s) for a in v)
    draw.text(S(30, 18), 'Reach: from a spoken request to a guided touch', font=font(30, True), fill=t['text'])
    for x, y, w, h, title, colour in GROUPS:
        draw.rounded_rectangle(S(x, y, x+w, y+h), radius=round(14*s), fill=t['panel'], outline=colour, width=max(2, round(2*s)))
        draw.text(S(x+16, y+12), title, font=font(20, True), fill=colour)
    import math

    def head(tail, tip):
        angle = math.atan2(tip[1]-tail[1], tip[0]-tail[0])
        left = (tip[0]-11*math.cos(angle-.45), tip[1]-11*math.sin(angle-.45))
        right = (tip[0]-11*math.cos(angle+.45), tip[1]-11*math.sin(angle+.45))
        draw.polygon([S(*tip), S(*left), S(*right)], fill=t['line'])
    for points, label, both, (lx, ly) in segments():
        draw.line([S(*p) for p in points], fill=t['line'], width=max(2, round(2*s)), joint='curve')
        head(points[-2], points[-1])
        if both:
            head(points[1], points[0])
        if label:
            f = font(14)
            mx, my = lx, ly+2
            w = draw.textlength(label, font=f)/s
            draw.rectangle(S(mx-w/2-3, my-14, mx+w/2+3, my+3), fill=t['panel'])
            draw.text(S(mx-w/2, my-15), label, font=f, fill=t['muted'])
    for x, y, w, h, title, lines, colour in BOXES.values():
        draw.rounded_rectangle(S(x, y, x+w, y+h), radius=round(10*s), fill=t['box'], outline=colour, width=max(2, round(2.5*s)))
        draw.rectangle(S(x, y+3, x+8, y+h-3), fill=colour)
        draw.text(S(x+20, y+11), title, font=font(20, True), fill=t['text'])
        for i, line in enumerate(lines):
            draw.text(S(x+20, y+42+i*22), line, font=font(16), fill=t['muted'])
    draw.text(S(30, HEIGHT-26), 'Times are measured medians or 95th percentiles from our runs. Orange: Meta Model API. '
              'Blue: local. Green: devices.', font=font(15), fill=t['muted'])
    image.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--png', type=Path)
    parser.add_argument('--dark', action='store_true')
    args = parser.parse_args()
    if args.png:
        png(args.png, 'dark' if args.dark else 'light')
    else:
        (Path(__file__).resolve().parent/'architecture.svg').write_text(svg(), encoding='utf-8')
