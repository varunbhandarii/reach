"""Small drawing functions; all HighGUI calls remain on the main thread."""

import cv2

from .controller import target_center


def draw_overlay(image, hand, target, decision, stats, *, muted=False, trial=None, stale_box=None):
    canvas = image.copy()
    center = target_center(target)
    if center is not None:
        box = target.instances[target.chosen].box_xyxy
        color = (0, 220, 0) if target.source == "sam" else (220, 220, 0)
        cv2.rectangle(canvas, box[:2], box[2:], color, 2)
    elif stale_box:
        cv2.rectangle(canvas, stale_box[:2], stale_box[2:], (130, 130, 130), 1)
    if hand and hand.present and hand.fingertip_px:
        point = tuple(round(x) for x in hand.fingertip_px)
        cv2.circle(canvas, point, 6, (255, 0, 255), -1)
        if center:
            cv2.arrowedLine(canvas, point, tuple(round(x) for x in center), (0, 255, 255), 2)
    cue = decision.cue if decision else None
    rows = [f"{decision.state if decision else 'IDLE'} {'MUTED' if muted else ''}  {target.phrase if target else ''}",
            f"fps {stats.get('capture_fps', 0):.1f}  cost ~${stats.get('estimated_cost_usd', 0):.3f}",
            f"SAM age {target.age_ms:.0f} ms" if target else "No current target",
            f"pan {cue.pan:+.2f}  pitch {cue.freq_hz:.0f} Hz  pulse {cue.pulse_hz:.1f} Hz" if cue else "Guide silent",
            "1-4 object | click point | c clear | m mute | r recheck | q quit"]
    if stats.get("roll_warning"):
        rows.append("Keep phone level: camera roll exceeds 20 degrees")
    if stats.get("upside_down"):
        rows.append("Phone appears inverted: cue directions flipped (auto)")
    if stats.get("error"):
        rows.append(str(stats["error"])[:100])
    if trial:
        rows.extend(trial)
    for index, text in enumerate(rows):
        y = 20 + index*21
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
