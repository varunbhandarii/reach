"""Sharp, source-coordinate overlays with a visible representation of the sound."""

import math
import time

import cv2

from .controller import target_center


def sound_values(cue,now,enabled=True):
    if cue is None or not enabled:
        return {'active':False,'pan':0.,'pitch':.5,'pulse':False,'on_line':False}
    return {'active':True,'pan':max(-1.,min(1.,cue.pan)),
        'pitch':max(0.,min(1.,math.log(max(220.,min(880.,cue.freq_hz))/220)/math.log(4))),
        'pulse':bool(math.sin(2*math.pi*cue.pulse_hz*now)>0),'on_line':cue.noise>0}


def draw_overlay(image, hand, target, decision, stats, *, muted=False, trial=None, stale_box=None,scale=1,now=None):
    if scale not in (1,2):
        raise ValueError('Overlay scale must be one or two')
    canvas=cv2.resize(image,(image.shape[1]*scale,image.shape[0]*scale)) if scale!=1 else image.copy()
    width,height=canvas.shape[1]//scale,canvas.shape[0]//scale
    pt=lambda xy:tuple(round(v*scale) for v in xy)
    def text(value,xy,size=.8,color=(245,245,245)):
        cv2.putText(canvas,str(value),pt(xy),cv2.FONT_HERSHEY_SIMPLEX,size*scale,color,max(1,scale),cv2.LINE_AA)
    def box(bounds,color,thickness=2):
        cv2.rectangle(canvas,pt(bounds[:2]),pt(bounds[2:]),color,thickness*scale)
    center=target_center(target)
    if target:
        for index,instance in enumerate(target.instances):
            chosen=index==target.chosen
            box(instance.box_xyxy,(50,240,100) if chosen else (170,170,170),2 if chosen else 1)
            text(index+1,instance.box_xyxy[:2],.5)
    elif stale_box:
        box(stale_box,(120,120,120),1)
    for item in stats.get('excluded_instances',[]):
        x0,y0,x1,y1=item['box_xyxy']
        cv2.line(canvas,pt((x0,y0)),pt((x1,y1)),(60,60,255),2*scale)
        cv2.line(canvas,pt((x0,y1)),pt((x1,y0)),(60,60,255),2*scale)
    if hand and hand.present and hand.fingertip_px:
        cv2.circle(canvas,pt(hand.fingertip_px),6*scale,(255,80,255),-1)
        if center:
            cv2.arrowedLine(canvas,pt(hand.fingertip_px),pt(center),(40,240,255),2*scale)
    state=decision.state if decision else 'IDLE'
    cv2.rectangle(canvas,(0,0),pt((width,83)),(20,24,30),-1)
    text(state+('  MUTED' if muted else ''),(10,29),.8)
    voice=stats.get('voice',{})
    online='off' if voice.get('enabled') is False else 'online' if voice.get('connected') else 'connecting'
    text(f"{target.phrase if target else 'No target'} | Voice {online} | {stats.get('capture_fps',0):.0f} fps | ~${stats.get('estimated_cost_usd',0):.3f}",(10,52),.45)
    text('1-4 find | D describe | C cancel | V talk | M mute | Q quit',(10,73),.42)
    if trial:
        for i,line in enumerate(trial):
            y=104+i*23
            cv2.rectangle(canvas,pt((0,y-20)),pt((width,y+4)),(25,29,35),-1)
            text(line[:90],(10,y),.48)
    checked=stats.get('readback',{})
    result=checked.get('last')
    notice='Checking touched object...' if checked.get('busy') else ''
    if result and not checked.get('busy'):
        label=result.get('error') or ('match' if result.get('matches') is True else 'mismatch' if result.get('matches') is False else 'unsure')
        notice=f"Read-back: {result.get('object_name') or ''} {label} {(result.get('latency_ms') or 0):.0f} ms"
    if voice.get('ptt_active'):
        notice='Listening: tap V again to finish'
    if stats.get('error'):
        notice=str(stats['error'])[:85]
    if notice:
        y=106+(len(trial)*23 if trial else 0)
        cv2.rectangle(canvas,pt((0,y-19)),pt((width,y+5)),(25,29,35),-1)
        text(notice[:85],(10,y),.46)
    scene=stats.get('scene',{}).get('last') or {}
    objects=scene.get('objects',())
    if objects:
        y=105+(len(trial)*23 if trial else 0)
        for name,position in objects[:6]:
            cv2.rectangle(canvas,pt((width-230,y-18)),pt((width,y+4)),(25,29,35),-1)
            text(f'{name}: {position}'[:36],(width-224,y),.45)
            y+=23
    sound_top=max(84,height-116)
    caption=stats.get('speech',{}).get('text') or stats.get('speech',{}).get('caption') or ''
    if caption:
        # Text is drawn after image scaling, including caption wrapping.
        max_chars=max(12,int((width-20)/14))
        lines=[]
        for word in caption.split():
            if not lines or len(lines[-1])+len(word)+1>max_chars:
                lines.append(word)
            else:
                lines[-1]+=' '+word
        for i,line in enumerate(lines[-2:]):
            y=sound_top-12-(len(lines[-2:])-i-1)*30
            cv2.rectangle(canvas,pt((0,y-25)),pt((width,y+6)),(20,24,30),-1)
            text(line,(10,y),.8)
    cv2.rectangle(canvas,pt((0,sound_top)),pt((width,height)),(20,24,30),-1)
    enabled=not muted and stats.get('route_ok',True) and voice.get('stereo_ok',True)
    values=sound_values(decision.cue if decision else None,time.perf_counter() if now is None else now,enabled)
    left,right=20,max(120,width-200)
    y=sound_top+50
    cv2.line(canvas,pt((left,y)),pt((right,y)),(130,140,155),3*scale)
    cv2.line(canvas,pt(((left+right)/2,y-10)),pt(((left+right)/2,y+10)),(80,90,110),scale)
    marker=left+(values['pan']+1)/2*(right-left)
    cv2.circle(canvas,pt((marker,y)),8*scale,(70,230,150) if values['active'] else (100,100,100),-1)
    text('LEFT',(left,sound_top+25),.5)
    text('RIGHT',(right-58,sound_top+25),.5)
    text('Guide sound' if values['active'] else 'Guide silent',(left,sound_top+91),.6)
    px=width-144
    cv2.line(canvas,pt((px,sound_top+22)),pt((px,sound_top+89)),(140,150,160),2*scale)
    cv2.circle(canvas,pt((px,sound_top+89-67*values['pitch'])),5*scale,(100,220,255),-1)
    text('forward',(px+10,sound_top+27),.38)
    text('back',(px+10,sound_top+91),.38)
    cv2.circle(canvas,pt((width-30,sound_top+53)),10*scale,(100,240,180) if values['pulse'] else (55,65,75),-1)
    if values['on_line']:
        text('ON LINE',(left,sound_top+111),.46,(100,240,180))
    if not enabled:
        text('STEREO UNAVAILABLE' if not muted else 'MUTED',(left,sound_top+111),.46,(90,160,255))
    return canvas
