from dataclasses import replace
import numpy as np
from reach.contact import ContactRule
from reach.masks import encode_rle
from reach.types import HandObs, TargetObs, Instance


def target(box=(100,100,200,200), age=0, mask=None):
    return TargetObs(1,'wallet',(Instance('1',box,.9,mask),),0,'sam',age)


def hand(i, point=(150,150), hull=((120,120),(180,120),(180,180),(120,180))):
    return HandObs(i,True,point,None,hull,(0,0),1)


def test_still_inside_fires_once_and_duplicate_frames_do_not_count():
    rule=ContactRule();fired=[]
    for i in range(15):
        value=rule.update(i*.05,hand(i),target())
        fired.append(value.fired)
        assert not rule.update(i*.05,hand(i),target()).fired
    assert sum(fired)==1 and fired[5]


def test_fast_pass_and_stopped_beside_do_not_fire():
    for point in (lambda i:(90+i*10,150), lambda i:(250,150)):
        rule=ContactRule()
        assert not any(rule.update(i*.05,hand(i,point(i)),target()).fired for i in range(12))


def test_shared_camera_translation_and_thin_box_margin():
    rule=ContactRule()
    values=[]
    for i in range(10):
        b=(100+i*3,100+i*2,106+i*3,200+i*2)
        values.append(rule.update(i*.05,hand(i,(115+i*3,150+i*2)),target(b)))
    assert any(v.fired for v in values)


def test_fresh_mask_overlap_is_required_but_stale_masks_are_skipped():
    mask=encode_rle(np.ones((100,100),np.uint8))
    for age,expected in [(0,False),(2001,True)]:
        rule=ContactRule()
        values=[rule.update(i*.05,hand(i,hull=()),target(age=age,mask=mask)) for i in range(10)]
        assert any(v.fired for v in values)==expected
    rule=ContactRule()
    assert any(rule.update(i*.05,hand(i),target(mask=mask)).fired for i in range(10))


def test_reset_retains_refractory_and_absence_resets_dwell():
    rule=ContactRule()
    for i in range(8): rule.update(i*.05,hand(i),target())
    rule.reset()
    assert not any(rule.update(.5+i*.05,hand(i),target()).fired for i in range(10))
    rule.reset()
    assert any(rule.update(3+i*.05,hand(i),target()).fired for i in range(10))
    rule.reset();rule.update(4,hand(1),target())
    rule.update(4.1,replace(hand(2),present=False),target())
    assert not rule.update(4.2,hand(3),target()).still



def test_mask_decodes_once_per_target_update_and_rechecks_changed_mask(monkeypatch):
    from dataclasses import replace
    from reach.masks import encode_rle,decode_rle
    import numpy as np
    from reach.types import HandObs,Instance,TargetObs
    calls=[]
    def decode(value): calls.append(True);return decode_rle(value)
    monkeypatch.setattr('reach.contact.decode_rle',decode)
    rule=ContactRule()
    mask=encode_rle(np.ones((100,100),np.uint8))
    t=TargetObs(1,'wallet',(Instance('a',(100,100,200,200),.9,mask),),0,'sam',0)
    for i in range(15):
        h=HandObs(i,True,(150,150),None,((100,100),(200,100),(200,200),(100,200)),(0,0),1)
        assert rule.update(i*.03,h,t).overlap==1
    assert len(calls)==1
    t=replace(t,frame_id=2)
    rule.update(.5,replace(h,frame_id=20),t)
    assert len(calls)==2
    t=replace(t,instances=(replace(t.instances[0],mask_rle=encode_rle(np.zeros((100,100),np.uint8))),))
    assert rule.update(.6,replace(h,frame_id=21),t).overlap==0
    assert len(calls)==3
