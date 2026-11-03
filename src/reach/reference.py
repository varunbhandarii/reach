"""Conservative pre-reach reference capture; never refresh through hand occlusion."""

from .spark import make_crop


def clean_target_box(hand,box):
    x0,y0,x1,y1=box
    if hand and hand.present:
        if hand.fingertip_px:
            x,y=hand.fingertip_px
            dx,dy=(x1-x0)*.25,(y1-y0)*.25
            if x0-dx<=x<=x1+dx and y0-dy<=y<=y1+dy:
                return False
        if hand.hull_px:
            xs,ys=zip(*hand.hull_px)
            if max(xs)>=x0 and min(xs)<=x1 and max(ys)>=y0 and min(ys)<=y1:
                return False
    return True


def capture_reference(camera,target,hand,now,previous=None,long_side=512):
    if target is None or target.chosen is None or not 0<=target.chosen<len(target.instances):
        return previous
    item=target.instances[target.chosen]
    # Detection IDs may change on a fresh query of the same requested object.
    # Explicit phrase/instance/exclusion effects reset the cache in the runner.
    # Preserve the last clear view through a refresh while the hand occludes it.
    if previous and previous['phrase']!=target.phrase:
        previous=None
    if previous and now-previous['t_mono']<.5:
        return previous
    if hand is None or hand.frame_id!=target.frame_id:
        return previous
    if not clean_target_box(hand,item.box_xyxy):
        return previous
    image=camera.get(target.frame_id)
    if image is None:
        return previous
    return {'jpeg':make_crop(image,item.box_xyxy,(),long_side),'frame_id':target.frame_id,
        'sam_id':item.sam_id,'box':tuple(item.box_xyxy),'phrase':target.phrase,'t_mono':now}
