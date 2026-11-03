from dataclasses import replace
import math

import numpy as np
import pytest

from reach.overlay import draw_overlay,sound_values
from reach.types import CueParams,Decision


def test_sound_strip_pan_log_pitch_pulse_and_mute():
    cue=CueParams(-1,220,2,False,.2,.2)
    v=sound_values(cue,.125)
    assert v['pan']==-1 and v['pitch']==0 and v['pulse'] and v['on_line']
    assert sound_values(replace(cue,freq_hz=440),.375)['pitch']==pytest.approx(.5)
    assert not sound_values(cue,.375)['pulse']
    assert sound_values(replace(cue,freq_hz=880),0)['pitch']==1
    assert not sound_values(cue,.125,False)['active']


def test_scaled_overlay_draws_after_resize_and_keeps_input_unchanged():
    source=np.full((480,640,3),70,np.uint8)
    decision=Decision(0,1,'GUIDING',0,0,.1,CueParams(.5,660,4,False,0,.2),None,None)
    stats={'speech':{'caption':'Found the wallet.'},'voice':{'connected':True}}
    one=draw_overlay(source,None,None,decision,stats,now=.1)
    two=draw_overlay(source,None,None,decision,stats,scale=2,now=.1)
    assert one.shape==(480,640,3) and two.shape==(960,1280,3)
    assert (source==70).all()
    # Sharp text is drawn in the large raster rather than interpolated from the small one.
    import cv2
    assert not np.array_equal(two,cv2.resize(one,(1280,960)))
