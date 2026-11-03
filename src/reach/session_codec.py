"""Plain JSON snapshots for deterministic session replay, with no device access."""

from dataclasses import fields, is_dataclass
import math

from .types import Command, HandObs, Instance, ReadBack, SceneList, TargetObs


def plain(value):
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, 'item'):
        return plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError('Session snapshots require finite, plain JSON values')


def replay_config(cfg, objects=None):
    """Only reducer settings: excludes keys, endpoints and device identities."""
    brain_keys=('prediction','veto_s','search_prompt_s','every_s','give_up_s','hand_lost_speak_s',
                'target_lost_s','target_lost_speak_s','target_give_up_s','network_speak_every_s',
                'readback_wait_s','mismatch_confidence','disambiguate_s','ask_touch_enabled','ask_touch_s')
    contact_keys=('enabled','still_ms','refractory_s','box_grow','min_margin_px','still_speed','hold_frames','overlap_min')
    return plain({'brain': {key:value for key,value in cfg.get('brain',{}).items() if key in brain_keys},
                  **({'scene': {'deadline_s': cfg['scene'].get('deadline_s', 6.0)}} if 'scene' in cfg else {}),
                  **({'safety': cfg['safety']} if 'safety' in cfg else {}),
                  'contact': {key:value for key,value in cfg.get('contact',{}).items() if key in contact_keys},
                  'audio': {'gain': cfg.get('audio', {}).get('gain', .2)},
                  'sam': {'tie_margin': cfg.get('sam', {}).get('tie_margin', .15)},
                  'spark': {key: cfg.get('spark', {}).get(key, default)
                            for key, default in [('enabled', True), ('deadline_s', 6.0)]},
                  'prediction': cfg.get('prediction', .5),
                  'objects': list(objects if objects is not None else cfg.get('objects', ['wallet','pen','ball','apple']))})


def restore_inputs(data):
    hand = data.get('hand')
    if hand is not None:
        hand = dict(hand)
        for key in ('fingertip_px', 'palm_px', 'velocity_px_s'):
            if hand.get(key) is not None:
                hand[key] = tuple(hand[key])
        for key in ('hull_px', 'landmarks_px'):
            hand[key] = tuple(tuple(point) for point in hand.get(key, ()))
        hand = HandObs(**hand)
    target = data.get('target')
    if target is not None:
        target = dict(target)
        target['instances'] = tuple(Instance(**{**row, 'box_xyxy': tuple(row['box_xyxy'])})
                                    for row in target['instances'])
        target = TargetObs(**target)
    return dict(hand=hand, target=target, perception_stats=data.get('perception_stats', {}),
                **({'frame_id': data['frame_id']} if 'frame_id' in data else {}),
                commands=tuple(Command(**row) for row in data.get('commands', ())),
                results=tuple(SceneList(**{**row,'objects':tuple(tuple(item) for item in row['objects'])})
                              if 'objects' in row else ReadBack(**row) for row in data.get('results', ())),
                speaking=data.get('speaking', False), frame_size=tuple(data.get('frame_size', (640,480))),
                flipped=data.get('flipped', False))
