from dataclasses import replace
import math

import cv2
import numpy as np
import pytest

from reach.overlay import (Captions, Screen, TextRenderer, chips, draw_overlay, layout_for, sound_values,
                           state_card, stereo_levels, BG, TEXT, MUTED, CARD)
from reach.types import CueParams, Decision, HandObs, Instance, TargetObs


def test_sound_strip_pan_log_pitch_pulse_and_mute():
    cue=CueParams(-1,220,2,False,.2,.2)
    v=sound_values(cue,.125)
    assert v['pan']==-1 and v['pitch']==0 and v['pulse'] and v['on_line']
    assert sound_values(replace(cue,freq_hz=440),.375)['pitch']==pytest.approx(.5)
    assert not sound_values(cue,.375)['pulse']
    assert sound_values(replace(cue,freq_hz=880),0)['pitch']==1
    assert not sound_values(cue,.125,False)['active']


@pytest.mark.parametrize('hz', [2, 6, 12])
def test_pulse_indicator_blinks_at_the_cue_rate(hz):
    cue=CueParams(0,440,hz,False,0,.2)
    samples=[sound_values(cue,t/1000)['pulse'] for t in range(0,2000)]
    rises=sum(1 for a,b in zip(samples,samples[1:]) if b and not a)
    assert abs(rises-2*hz) <= 1


def test_stereo_bars_follow_the_equal_power_pan_law():
    assert stereo_levels(-1)==pytest.approx((1,0),abs=1e-9)
    assert stereo_levels(0)==pytest.approx((math.sqrt(.5),math.sqrt(.5)))
    assert stereo_levels(1)==pytest.approx((0,1),abs=1e-9)


def test_canvas_is_16_by_9_and_input_is_unchanged():
    source=np.full((480,640,3),70,np.uint8)
    decision=Decision(0,1,'GUIDING',0,0,.1,CueParams(.5,660,4,False,0,.2),None,None)
    stats={'speech':{'text':'Found the wallet.'},'voice':{'connected':True}}
    one=draw_overlay(source,None,None,decision,stats,now=.1)
    two=draw_overlay(source,None,None,decision,stats,scale=2,now=.1)
    assert one.shape==(720,1280,3) and two.shape==(1080,1920,3)
    assert (source==70).all()


def test_panels_never_cover_the_camera_image():
    rng=np.random.default_rng(3)
    source=rng.integers(0,255,(480,640,3),dtype=np.uint8)
    decision=Decision(0,1,'SEARCHING',None,None,None,None,'Looking for the wallet.',None)
    stats={'speech':{'text':'Looking for the wallet.'},'voice':{'connected':True},'phrase':'wallet'}
    for scale,mode in ((1,'presentation'),(2,'presentation'),(2,'debug')):
        layout=layout_for(640,480,scale)
        canvas=Screen().draw(source,None,None,decision,stats,scale=scale,now=1,mode=mode)
        x,y,w,h=layout.image
        expected=cv2.resize(source,(w,h),interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(canvas[y:y+h,x:x+w],expected)


def test_layout_maps_clicks_back_to_source_pixels():
    layout=layout_for(640,480,2)
    assert layout.to_source(layout.to_canvas((123,321)))==pytest.approx((123,321),abs=.5)
    assert layout.to_source((layout.pane_width+10,100)) is None
    wide=layout_for(1280,720,1)
    assert wide.image[3] < wide.height and wide.to_source((wide.image[0]+1, wide.image[1]+1)) is not None


def test_captions_expire_and_clear_on_a_state_change():
    captions=Captions()
    captions.update(0,heard='find the pen',speaking='Looking for the pen.',state='SEARCHING')
    assert captions.visible(1)['heard'][0]=='find the pen'
    captions.update(1.5,heard='find the pen',speaking=None,state='SEARCHING')
    assert captions.visible(3)['said'][0]=='Looking for the pen.'
    assert captions.visible(3.6)['said'] is None
    assert captions.visible(4.2)['heard'] is None
    captions.update(5,speaking='Bring your hand into view.',state='HAND_LOST')
    captions.update(6,speaking=None,state='HAND_LOST')
    captions.update(6.2,speaking=None,state='GUIDING')
    assert captions.visible(6.3)['said'] is None


def test_plain_language_state_card():
    stats={'voice':{'connected':True},'phrase':'pen'}
    decision=lambda state: Decision(0,1,state,None,None,None,None,None,None)
    assert state_card(decision('GUIDING'),None,stats)[0]=='Guiding to the pen'
    assert state_card(decision('HAND_LOST'),None,stats)[0]=='Hand out of view'
    done={**stats,'readback':{'last':{'matches':False}}}
    assert state_card(decision('DONE'),None,done)[0]=='Not the pen'
    lost={**stats,'voice':{'connected':True,'stereo_ok':False}}
    assert state_card(decision('GUIDING'),None,lost)[0].startswith('Stereo lost')
    assert [c[0] for c in chips(stats)]==['Voice','Stereo','Camera']
    paused={**stats,'camera_ok':False}
    assert state_card(decision('GUIDING'),None,paused)[0]=='Camera paused, reconnecting'
    assert chips(paused,camera_ok=False)[2][2]=='no frames'


def contrast(a,b):
    def lum(c):
        def ch(v):
            v/=255
            return v/12.92 if v<=.03928 else ((v+.055)/1.055)**2.4
        r,g,bl=(ch(v) for v in c)
        return .2126*r+.7152*g+.0722*bl
    hi,lo=sorted((lum(a),lum(b)),reverse=True)
    return (hi+.05)/(lo+.05)


def test_panel_text_meets_enhanced_contrast():
    for color in (TEXT,MUTED):
        for background in (BG,CARD):
            assert contrast(color,background) >= 7


def test_bundled_fonts_load_and_fall_back_cleanly(tmp_path):
    renderer=TextRenderer()
    assert renderer.available
    assert renderer.width('Guiding to the pen',40) > 0
    assert renderer.width('कलम खोज रही हूँ',30) > 0
    missing=TextRenderer(tmp_path)
    assert not missing.available
    source=np.zeros((480,640,3),np.uint8)
    canvas=Screen(missing).draw(source,None,None,None,{'voice':{}},now=0)
    assert canvas.shape==(720,1280,3)


def test_marks_sound_ring_and_low_vision_view_draw():
    source=np.full((480,640,3),90,np.uint8)
    target=TargetObs(1,'wallet',(Instance('a',(300,200,380,280),.9,None),Instance('b',(100,100,140,150),.8,None)),0,'sam',0)
    hand=HandObs(1,True,(200,300),(200,300),(),(0,0),1)
    decision=Decision(0,1,'GUIDING',.2,-.1,.25,CueParams(.5,600,6,False,.1,.2),None,None)
    stats={'voice':{'connected':True},'phrase':'wallet'}
    screen=Screen()
    plain=screen.draw(source,None,None,None,stats,now=0)
    marked=screen.draw(source,hand,target,decision,stats,now=1/24)
    assert not np.array_equal(plain[:, :960],marked[:, :960])
    low=screen.draw(source,hand,target,decision,stats,now=1/24,low_vision=True)
    assert low[:, :960].mean() < marked[:, :960].mean()


def test_preview_renders_every_sample_screen(tmp_path):
    from reach.ui_preview import render, scenarios
    written,(p50,p95)=render(tmp_path,scales=(1,),modes=('presentation',),timing=5)
    assert len(written)==len(scenarios())+1  # plus one low-vision view
    assert all(path.stat().st_size>10_000 for path in written)
    assert p50 is not None and p95 >= p50


def test_setup_writes_local_choices_and_refuses_airpods_microphones(tmp_path):
    import yaml
    from reach.setup_wizard import run_setup
    path=tmp_path/'config.local.yaml'
    path.write_text('hand:\n  model_path: models/hand_landmarker.task\n',encoding='utf-8')
    answers=iter(['2','1','2'])
    run_setup(path,input_fn=lambda prompt: next(answers),cameras=[(0,'1280x720'),(1,'640x480')],
              audio=(['Headphones'],['MacBook Air Microphone','AirPods Pro']))
    saved=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert saved['camera']['device_index']==1 and saved['audio']['output_device']=='Headphones'
    assert 'voice' not in saved and saved['hand']['model_path']=='models/hand_landmarker.task'
    answers=iter(['http://192.168.1.5:8080/video','','1'])
    run_setup(path,input_fn=lambda prompt: next(answers),cameras=[],audio=(['Headphones'],['MacBook Air Microphone']))
    saved=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert saved['camera']['source']=='android_mjpeg' and saved['voice']['input_device']=='MacBook Air Microphone'


def test_start_is_handheld_run():
    from reach.cli import parser
    import reach.cli as cli
    seen={}
    class Stop(Exception): pass
    def fake_configure(args):
        seen['args']=args
        raise ValueError('stop here')
    original=cli.configure
    cli.configure=fake_configure
    try:
        assert cli.main(['start','--duration','1'])==2
    finally:
        cli.configure=original
    assert seen['args'].command=='run' and seen['args'].mount=='handheld' and seen['args'].duration==1


def test_view_keys_and_clicks_use_the_new_layout():
    from types import SimpleNamespace
    from reach.runner import Runner
    from reach.types import Frame
    events,clicks=[],[]
    fake=SimpleNamespace(log=SimpleNamespace(emit=lambda kind,data: events.append((kind,data))),_trial=None,
                         _ui_mode='debug',_low_vision=False,args=SimpleNamespace(overlay_scale=2),
                         _last_frame=Frame(1,0,0,640,480,'x','file'),audio=SimpleNamespace(set=lambda cue: None),
                         select_point=lambda x,y: clicks.append((x,y)))
    Runner._key(fake,ord('u'),None)
    Runner._key(fake,ord('l'),None)
    assert fake._ui_mode=='presentation' and fake._low_vision
    assert [d for k,d in events if k=='ui_mode'][-1]=={'mode':'presentation','low_vision':True}
    layout=layout_for(640,480,2)
    Runner._mouse(fake,cv2.EVENT_LBUTTONDOWN,*layout.to_canvas((320,240)),0,None)
    Runner._mouse(fake,cv2.EVENT_LBUTTONDOWN,layout.pane_width+20,100,0,None)
    assert clicks==[pytest.approx((320,240),abs=1)]
