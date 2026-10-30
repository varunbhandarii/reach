"""Deterministic box, relative-motion and optional mask contact heuristic."""

from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from .controller import target_center
from .masks import decode_rle
from .tracker import hand_mask


@dataclass(frozen=True)
class ContactState:
    fired: bool = False
    inside: bool = False
    still: bool = False
    overlap: float | None = None
    rel_speed: float | None = None
    hold_frames: int = 0


class ContactRule:
    def __init__(self, cfg=None):
        self.cfg = (cfg or {}).get('contact', {})
        self.window_s = float(self.cfg.get('still_ms', 150))/1000
        self.refractory_s = float(self.cfg.get('refractory_s', 2))
        if self.window_s <= 0 or self.refractory_s < 0:
            raise ValueError('invalid contact timing')
        self._last_fire = None
        self.reset()

    def reset(self):
        self._samples = deque()
        self._hold = 0
        self._frame_id = None
        self._fired = False
        self._last_time = None
        self._state = ContactState()

    def update(self, now, hand, target, mask=None):
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('contact timestamps must be finite and ordered')
        self._last_time = now
        center = target_center(target)
        if hand is None or not hand.present or hand.fingertip_px is None or center is None:
            self._samples.clear(); self._hold = 0; self._frame_id = None
            self._state = ContactState()
            return self._state
        if hand.frame_id == self._frame_id:
            return ContactState(False, self._state.inside, self._state.still,
                                self._state.overlap, self._state.rel_speed, self._hold)
        self._frame_id = hand.frame_id
        x0,y0,x1,y1 = target.instances[target.chosen].box_xyxy
        width, height = x1-x0, y1-y0
        grow = float(self.cfg.get('box_grow', .2)); margin = float(self.cfg.get('min_margin_px', 12))
        mx, my = max(margin, width*grow), max(margin, height*grow)
        fx, fy = hand.fingertip_px
        if not all(math.isfinite(v) for v in (fx,fy)):
            self._samples.clear(); self._hold = 0
            return ContactState()
        inside = x0-mx <= fx <= x1+mx and y0-my <= fy <= y1+my
        vector = (np.asarray([fx,fy])-center)/math.hypot(width,height)
        if not inside:
            self._samples.clear()
        else:
            self._samples.append((now, vector))
        # Keep one anchor on/before the window boundary.
        while len(self._samples)>2 and self._samples[1][0] <= now-self.window_s:
            self._samples.popleft()
        speed = None
        if len(self._samples)>1:
            dt = now-self._samples[0][0]
            if dt > 0:
                speed = sum(float(np.linalg.norm(b[1]-a[1])) for a,b in zip(self._samples, list(self._samples)[1:]))/dt
        still = (inside and len(self._samples)>1 and now-self._samples[0][0]+1e-9 >= self.window_s
                 and speed is not None and speed < float(self.cfg.get('still_speed', .25)))
        overlap = None
        instance = target.instances[target.chosen]
        rle = mask if mask is not None else instance.mask_rle
        if rle is not None and target.age_ms <= 2000:
            raster = decode_rle(rle) if isinstance(rle, dict) else np.asarray(rle)
            if raster.shape != (height,width):
                raise ValueError('contact mask must be box-local')
            canvas = hand_mask((max(y1,1),max(x1,1)), hand)[y0:y1,x0:x1]
            overlap = float(np.count_nonzero((canvas!=0)&(raster!=0)))/max(1,np.count_nonzero(raster))
        qualifies = still and (overlap is None or overlap >= float(self.cfg.get('overlap_min', .1)))
        self._hold = self._hold+1 if qualifies else 0
        fired = (self._hold >= int(self.cfg.get('hold_frames',3)) and not self._fired
                 and (self._last_fire is None or now-self._last_fire+1e-9 >= self.refractory_s))
        if fired:
            self._fired = True; self._last_fire = now
        self._state = ContactState(fired, inside, still, overlap, speed, self._hold)
        return self._state
