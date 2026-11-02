"""Small drawing functions; all HighGUI calls remain on the main thread."""

import cv2

from .controller import target_center


def draw_overlay(image, hand, target, decision, stats, *, muted=False, trial=None, stale_box=None):
    canvas = image.copy()
    center = target_center(target)
    if target:
        for index,instance in enumerate(target.instances):
            if index != target.chosen:
                cv2.rectangle(canvas,instance.box_xyxy[:2],instance.box_xyxy[2:],(150,150,150),1)
                cv2.putText(canvas,str(index+1),instance.box_xyxy[:2],cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    for excluded in stats.get('excluded_instances',[]):
        x0,y0,x1,y1 = excluded['box_xyxy']
        cv2.line(canvas,(x0,y0),(x1,y1),(0,0,255),2)
        cv2.line(canvas,(x0,y1),(x1,y0),(0,0,255),2)
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
            "1-4 find | C cancel | R recheck | V talk | Y/N | M mute | Q quit"]
    voice = stats.get('voice', {})
    speech = stats.get('speech', {})
    contact = stats.get('contact', {})
    rows.append('Voice '+('off' if voice.get('enabled') is False else 'online' if voice.get('connected') else 'offline')
                + (' | tap V again to finish' if voice.get('ptt_active') else ''))
    if voice.get('stereo_ok') is False:
        rows.append('STEREO UNAVAILABLE: guidance muted; check audio routing')
    if voice.get('last_transcript'):
        rows.append('Heard: '+voice['last_transcript'][:75])
    caption = speech.get('text') or speech.get('caption')
    if caption:
        rows.append('Speech: '+caption[:75])
    if speech.get('error'):
        rows.append('Speech unavailable; captions only')
    if contact:
        rows.append(f"Contact: inside={contact.get('inside')} still={contact.get('still')} overlap={contact.get('overlap')}")
    readback = stats.get('readback',{})
    result = readback.get('last')
    if readback.get('busy'):
        rows.append('Checking touched object...')
    elif result:
        label = result.get('error') or ('match' if result.get('matches') is True else 'mismatch' if result.get('matches') is False else 'unsure')
        rows.append(f"Read-back: {result.get('object_name') or ''} {label} {result.get('latency_ms',0):.0f} ms")
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
