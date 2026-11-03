"""Deterministic session reducer; all time and observations come from the caller.

The hand-thread owner calls step every hand frame and on no-frame ticks.
Pass hand=None when its observation expires. perception_stats may include
voice_online=False (or voice_offline=True) for the once-per-run voice notice.
Effects use the existing log type/data envelope and perform no I/O here.
"""

from dataclasses import asdict, replace
import math
from .contact import ContactRule, ContactState
from .controller import target_center, predict_tip, map_vector
from .types import Decision, Effect, ReadBack, SceneList
from .session_codec import plain, replay_config
from .safety import blocked_term, REFUSAL
from .config import scene_deadline


GREETING = "Reach is ready. Say find and an object, or what's on the table."
HELP = "Say find and an object. What's on the table. Stop. Repeat. Louder or quieter."
# Replies to an explicit request are spoken even in brief mode.
ANSWER_KEYS = frozenset(('help', 'louder', 'quieter', 'loudest', 'quietest', 'brief_on', 'brief_off', 'yes_ack', 'no_ack'))
# Queued prompts that no longer apply after a transition; a playing phrase finishes.
STALE_PROMPTS = {('HAND_LOST', 'GUIDING'): ('hand_lost',), ('TARGET_LOST', 'GUIDING'): ('target_lost',),
                 ('SEARCHING', 'GUIDING'): ('sweep', 'looking'), ('SEARCHING', 'DISAMBIGUATING'): ('sweep', 'looking')}


class Brain:
    def __init__(self, cfg=None, objects=('wallet','pen','ball','apple')):
        self.cfg = cfg or {}
        self._scene_deadline = scene_deadline(self.cfg)
        self.params = self.cfg.get('brain', {})
        self.objects = tuple(objects)
        self.contact = ContactRule(self.cfg)
        self.state = 'IDLE'
        self.phrase = None
        self.entered = 0.0
        self._seq = 0
        self._last_now = None
        self._search_start = self._last_prompt = self._target_missing = None
        self._hand_notice = self._target_notice = self._many_notice = False
        self._voice_notice = False
        self._last_network = None
        self._last_spoken = None
        self._last_contact = None
        self.contact_state = None
        self._request_id = 0
        self._pending = self._disambiguation = None
        self._fallback_spoken = self._choice_prompted = False
        self._mismatches = 0
        self._ask_started = None
        self._ask_until = None
        self._ask_asked = False
        self._ask_suppressed_until = 0.0
        self._scene_pending = None
        self.last_scene = ()
        self.language = 'en'
        self.spoken_phrase = None
        self._safety_blocked = False
        self.volume_db = 0.0
        self.brief = False
        self._greeted = False

    def _volume(self, kind, effects):
        step = self._number('volume_step_db', 3)
        low, high = self._number('volume_min_db', -12), self._number('volume_max_db', 6)
        level = self.volume_db+(step if kind == 'louder' else -step)
        if level > high+1e-9 or level < low-1e-9:
            self._speak(effects, "That's the loudest." if kind == 'louder' else "That's the quietest.",
                        'loudest' if kind == 'louder' else 'quietest', 1)
            return
        self.volume_db = level
        effects.append(Effect('volume', payload={'db': level}))
        effects.append(Effect('log', key='volume', payload={'db': level}))
        self._speak(effects, 'Louder.' if kind == 'louder' else 'Quieter.', kind, 1)

    def _number(self, key, default):
        return float(self.params.get(key, default))

    def _speak(self, effects, text, key, priority=1):
        if self.brief and priority == 0 and key not in ANSWER_KEYS:
            return
        self._last_spoken = (text,key,priority)
        payload = ({'language': self.language, 'spoken_phrase': self.spoken_phrase}
                   if self.language != 'en' else None)
        effects.append(Effect('speak', text, key, priority, payload))

    def _command_language(self, command):
        # The reducer is platform-independent; the speaker keeps Windows English.
        self.language = 'hi' if (command.language or 'en').split('-')[0].casefold() == 'hi' else 'en'
        self.spoken_phrase = command.spoken_phrase

    def _name(self):
        return f'the {self.phrase}' if self.phrase else 'the selected point'

    def _transition(self, now, state, effects, reason, *, force=False):
        old = self.state
        if old == state and not force:
            return
        rate = lambda s: 'IDLE' if s in ('IDLE','CONTACT','CONFIRMING','DONE','DESCRIBING') else 'GUIDING' if s=='DISAMBIGUATING' else s
        self.state, self.entered = state, now
        effects.append(Effect('log', key='state', payload={'from':old,'to':state,'reason':reason,'phrase':self.phrase}))
        if rate(old) != rate(state) or force:
            effects.append(Effect('perception_state', key=rate(state)))
        stale = STALE_PROMPTS.get((old, state)) if self.params.get('cancel_stale_prompts', False) else None
        if stale:
            effects.append(Effect('cancel_speech', payload={'keys': list(stale)}))
        if state == 'HAND_LOST': self._hand_notice = False
        if state == 'TARGET_LOST': self._target_notice = False

    def _clear(self, now, effects, reason):
        self._transition(now,'IDLE',effects,reason)
        self.phrase = None
        self._target_missing = None
        self._search_start = None
        self._pending = self._disambiguation = None
        self._scene_pending = None
        self._ask_until = self._ask_started = None
        self.contact.reset()
        effects.append(Effect('set_phrase', text=None))

    def _find(self, now, phrase, effects):
        self._safety_blocked = False
        self._scene_pending = None
        if (self.phrase == phrase and self.state in ('SEARCHING','GUIDING')):
            effects.append(Effect('force_query'))
            self._speak(effects,'Checking.','checking',0)
            return
        self.phrase = phrase
        self._search_start = now
        self._last_prompt = None
        self._target_missing = None
        self._many_notice = False
        self._last_contact = None
        self._pending = self._disambiguation = None
        self._mismatches = 0
        self._choice_prompted = self._ask_asked = False
        self._ask_started = self._ask_until = None
        self.contact.reset()
        effects.append(Effect('cancel_speech'))
        effects.append(Effect('set_phrase', text=phrase))
        self._transition(now,'SEARCHING',effects,'find',force=True)
        self._speak(effects,f'Looking for {self._name()}.','looking')

    def _where(self, hand, center, size, flipped):
        point = hand.fingertip_px if hand and hand.present and hand.fingertip_px else (size[0]/2,size[1]/2)
        dx,dy=(center[0]-point[0])/size[0],(center[1]-point[1])/size[1]
        if flipped: dx,dy=-dx,-dy
        side = 'left' if dx < -.05 else 'right' if dx > .05 else None
        depth = 'ahead' if dy < -.05 else 'back' if dy > .05 else None
        if self.params.get('direction_words', 'legacy') == 'plain':
            # Image down is toward the body for a phone held over the table.
            distance = {'ahead': 'farther away', 'back': 'closer to you'}.get(depth)
            if side and distance: return f'on your {side}, {distance}'
            if side: return f'to your {side}'
            if depth == 'ahead': return 'straight ahead'
            if depth == 'back': return 'closer to you'
            return 'right by your hand'
        if depth and side: return f'{depth} on your {side}'
        if side: return f'to your {side}'
        if depth=='back': return 'back toward you'
        return 'straight ahead'

    def _eligible(self, target):
        if target is None or target.chosen is None or not target.instances:
            return ()
        top = max((instance.conf or 0) for instance in target.instances)
        margin = float(self.cfg.get('sam', {}).get('tie_margin', .15))
        return tuple(instance for instance in target.instances if top-(instance.conf or 0) <= margin+1e-9)

    def _pick(self, target, word, hand, flipped, *, nearest=False):
        instances = self._eligible(target) or (target.instances if target else ())
        if not instances:
            return None
        center = lambda item: ((item.box_xyxy[0]+item.box_xyxy[2])/2,
                               (item.box_xyxy[1]+item.box_xyxy[3])/2)
        direction = -1 if flipped else 1
        if nearest:
            if hand and hand.present and hand.fingertip_px:
                return min(instances, key=lambda item: math.dist(center(item), hand.fingertip_px))
            return target.instances[target.chosen] if target.chosen is not None else instances[0]
        if word in ('left','right'):
            if len(instances)>1 and max(center(item)[0] for item in instances)-min(center(item)[0] for item in instances)<=1:
                return None
            return min(instances, key=lambda item: direction*center(item)[0]*(1 if word=='left' else -1))
        if word in ('farther','ahead','closer'):
            if len(instances)>1 and max(center(item)[1] for item in instances)-min(center(item)[1] for item in instances)<=1:
                return None
            return min(instances, key=lambda item: direction*center(item)[1]*(-1 if word=='closer' else 1))
        if word == 'other' and len(instances)>1:
            current = target.instances[target.chosen].sam_id if target.chosen is not None else None
            return next((item for item in instances if item.sam_id != current), None)
        return None

    def _select(self, now, target, instance, effects, words):
        effects.append(Effect('choose_instance', payload={'sam_id': instance.sam_id}))
        self.contact.reset()
        self._target_missing = None
        self._transition(now, 'GUIDING', effects, 'instance_choice')
        self._speak(effects, words, 'choice', 1)
        self._disambiguation = None
        return replace(target, chosen=next(i for i,item in enumerate(target.instances) if item.sam_id==instance.sam_id))

    def _begin_confirmation(self, now, target, effects, reason='contact'):
        selected = target.instances[target.chosen]
        self._request_id += 1
        self._pending = {'request_id': self._request_id, 'frame_id': target.frame_id,
                         'phrase': self.phrase, 'target': target, 'started': now}
        self._fallback_spoken = False
        self._transition(now, 'CONFIRMING', effects, reason)
        effects.append(Effect('earcon', key='contact'))
        if self.cfg.get('spark', {}).get('enabled', True) and self.phrase and self.phrase != 'clicked point':
            effects.append(Effect('readback', payload={'request_id': self._request_id,
                'frame_id': target.frame_id, 'box_xyxy': selected.box_xyxy, 'phrase': self.phrase}))
        else:
            self._touching(effects)
            self._finish_confirmation(now, effects, 'readback_disabled')

    def _touching(self, effects):
        if not self._fallback_spoken:
            self._speak(effects, f"You're touching {self._name()}.", 'contact', 2)
            self._fallback_spoken = True

    def _finish_confirmation(self, now, effects, reason):
        self._pending = None
        self._transition(now, 'DONE', effects, reason)
        effects.append(Effect('set_phrase', text=None))

    def _readback_results(self, now, results, effects):
        changed = False
        for result in results:
            if not isinstance(result, ReadBack):
                continue
            pending = self._pending
            matches = (self.state=='CONFIRMING' and pending is not None and
                       result.request_id==pending['request_id'] and result.target_frame_id==pending['frame_id']
                       and result.phrase==pending['phrase'])
            elapsed = now-pending['started'] if pending else 0
            if not matches or elapsed > float(self.cfg.get('spark',{}).get('deadline_s',6)):
                effects.append(Effect('log', key='readback_ignore', payload={
                    'request_id': result.request_id, 'reason': 'stale_or_unmatched'}))
                continue
            confidence = result.confidence
            confident = (isinstance(confidence,(int,float)) and not isinstance(confidence,bool) and math.isfinite(confidence)
                         and self._number('mismatch_confidence',.6) <= confidence <= 1)
            if result.error or type(result.matches) is not bool or not confident:
                self._touching(effects)
                self._finish_confirmation(now, effects, 'readback_error' if result.error else 'readback_unsure')
            elif result.matches:
                if elapsed <= self._number('readback_wait_s',2.5) and not self._fallback_spoken:
                    self._speak(effects, f'You have {self._name()}.', 'confirmed', 2)
                    if isinstance(result.hint,str) and result.hint.strip():
                        self._speak(effects, ' '.join(result.hint.split()[:6])[:120], 'hint', 0)
                else:
                    self._touching(effects)
                self._finish_confirmation(now, effects, 'readback_match')
            else:
                self._mismatches += 1
                if self._mismatches >= 2:
                    name = self._name()
                    effects.append(Effect('cancel_speech'))
                    self._clear(now, effects, 'repeated_mismatch')
                    self._speak(effects, f"I can't confirm {name}.", 'unconfirmed', 2)
                else:
                    old = pending['target']
                    chosen = old.instances[old.chosen]
                    effects.append(Effect('cancel_speech'))
                    effects.append(Effect('earcon', key='wrong'))
                    self._speak(effects, f"That's not {self._name()}.", 'wrong_object', 2)
                    if isinstance(result.object_name,str) and result.object_name.strip():
                        name = ' '.join(result.object_name.split()[:3])[:80]
                        self._speak(effects, f'It looks like a {name}.', 'object_name', 0)
                    effects.append(Effect('exclude_instance', payload={
                        'frame_id': pending['frame_id'], 'box_xyxy': chosen.box_xyxy}))
                    alternatives = tuple(item for item in old.instances if item.sam_id!=chosen.sam_id)
                    self._pending = None
                    self._target_missing = None
                    self.contact.reset()
                    if alternatives:
                        effects.append(Effect('choose_instance', payload={'sam_id': alternatives[0].sam_id}))
                        self._transition(now, 'GUIDING', effects, 'wrong_object_alternative')
                    else:
                        self._search_start, self._last_prompt = now, None
                        effects.append(Effect('set_phrase', text=self.phrase))
                        self._transition(now, 'SEARCHING', effects, 'wrong_object_search')
            changed = True
        return changed

    def _scene_results(self, now, results, effects, flipped):
        changed = False
        for result in results:
            if not isinstance(result, SceneList):
                continue
            pending = self._scene_pending
            if (self.state != 'DESCRIBING' or pending is None
                    or result.request_id != pending['request_id']
                    or (pending['frame_id'] is not None and result.frame_id != pending['frame_id'])
                    or now-self.entered >= self._scene_deadline):
                effects.append(Effect('log', key='scene_ignore', payload={
                    'request_id': result.request_id, 'reason': 'stale_or_unmatched'}))
                continue
            self._scene_pending = None
            changed = True
            if result.error:
                self._speak(effects, "I couldn't look around just now.", 'describe_error')
            else:
                objects = []
                for name, position in result.objects[:6]:
                    name = ' '.join(name.split()[:3])[:35]
                    if not name or position not in ('left', 'middle', 'right'):
                        continue
                    if flipped:
                        position = {'left': 'right', 'right': 'left', 'middle': 'middle'}[position]
                    objects.append((name, position))
                self.last_scene = tuple(objects)
                safe, hazards = [], []
                for name, position in objects:
                    if blocked_term(name, self.cfg):
                        hazards.append(name)
                    else:
                        location = 'in the middle' if position == 'middle' else f'on your {position}'
                        safe.append(f'a {name} {location}')
                if not objects:
                    self._speak(effects, "I don't see anything I can name.", 'scene_empty')
                if hazards:
                    self._speak(effects, 'Careful, there is a '+' and a '.join(hazards)+'.', 'hazard', 2)
                for i in range(0, len(safe), 3):
                    items = safe[i:i+3]
                    text = ', '.join(items[:-1])+' and '+items[-1] if len(items)>1 else items[0]
                    self._speak(effects, f'I can see {text}.', 'scene_list', 1 if i == 0 else 0)
            self._transition(now, 'IDLE', effects, 'describe_result')
        return changed

    def step(self, now, *, hand=None, target=None, perception_stats=None, commands=(), speaking=False,
             frame_size=(640,480), flipped=False, results=(), frame_id=None):
        if not math.isfinite(now) or (self._last_now is not None and now < self._last_now):
            raise ValueError('session timestamps must be finite and ordered')
        if min(frame_size)<=0:
            raise ValueError('frame dimensions must be positive')
        self._last_now=now
        stats=perception_stats or {}
        commands=tuple(commands)
        results=tuple(results)
        observed_stats = {key: stats[key] for key in ('stale','ambiguous','voice_online','voice_offline','consecutive_errors','hand_held','speech_ready') if key in stats}
        snapshot = plain(dict(seq=self._seq, now=now, hand=hand, target=target, perception_stats=observed_stats,
                              commands=commands, results=results, speaking=speaking,
                              frame_size=frame_size, flipped=flipped))
        if frame_id is not None:
            snapshot['frame_id'] = frame_id
        effects=[Effect('log', key='brain_in', payload=snapshot)]
        if self.params.get('greeting', False) and not self._greeted and stats.get('speech_ready'):
            self._greeted = True
            self._speak(effects, GREETING, 'greeting', 1)
        # Stop dominates a queued batch so a cancelled search never reaches I/O.
        stopping=next((c for c in reversed(commands) if c.kind in ('stop','cancel','start_over')),None)
        if stopping:
            effects.append(Effect('cancel_speech'))
            veto = stopping.kind=='cancel' and self.state=='SEARCHING' and now-self.entered <= self._number('veto_s',1.2)+1e-9
            self._clear(now,effects,stopping.kind)
            text,key = ('Ready.','ready') if stopping.kind=='start_over' else ('Cancelled.','cancelled') if veto else ('Stopped.','stopped')
            self._speak(effects,text,key,2)
            return self._decision(now,hand,target,None,effects)
        blocked = next(((c, blocked_term(c.phrase, self.cfg)) for c in commands
                        if c.kind == 'find' and blocked_term(c.phrase, self.cfg)), None)
        if blocked:
            command, term = blocked
            self._command_language(command)
            effects.append(Effect('cancel_speech'))
            self._clear(now, effects, 'safety_block')
            self._safety_blocked = True
            effects.append(Effect('log', key='safety_block', payload={
                'phrase': command.phrase, 'term': term, 'source': command.source}))
            self._speak(effects, REFUSAL, 'safety_refusal', 2)
            return self._decision(now, hand, target, None, effects)
        for command in commands:
            if command.kind=='find' and command.phrase:
                phrase=command.phrase.strip()
                if len(phrase)<=120 and not any(c in phrase for c in ',;\n'):
                    self._command_language(command)
                    self._find(now,phrase,effects)
            elif command.kind == 'describe' and self.state != 'CONFIRMING':
                self._command_language(command)
                effects.append(Effect('cancel_speech'))
                self._clear(now, effects, 'describe')
                self._request_id += 1
                scene_frame = frame_id if frame_id is not None else hand.frame_id if hand else target.frame_id if target else None
                self._scene_pending = {'request_id': self._request_id, 'frame_id': scene_frame}
                self._transition(now, 'DESCRIBING', effects, 'describe')
                effects.append(Effect('describe', payload=dict(self._scene_pending)))
                self._speak(effects, 'Looking around.', 'describing', 0)
            elif command.kind=='no' and self.state=='SEARCHING' and now-self.entered<=self._number('veto_s',1.2)+1e-9:
                effects.append(Effect('cancel_speech'));self._clear(now,effects,'veto')
                self._speak(effects,'Cancelled.','cancelled',2)
            elif command.kind=='yes' and self.state=='GUIDING' and self._ask_until is not None and now<=self._ask_until and target_center(target) is not None:
                self._ask_until = None
                self._begin_confirmation(now, target, effects, 'user_yes')
            elif command.kind=='no' and self._ask_until is not None:
                self._ask_until = None
                self._ask_suppressed_until = now+5
                self._speak(effects,'I heard no.','no_ack',0)
            elif command.kind=='choose' and self.state=='DISAMBIGUATING':
                options = target if target and target.phrase==self.phrase and target_center(target) else self._disambiguation
                instance = self._pick(options, command.phrase, hand, flipped)
                if instance is not None:
                    text = ('Guiding to the one on your '+command.phrase+'.' if command.phrase in ('left','right')
                            else 'Guiding to the '+('farther' if command.phrase=='ahead' else command.phrase)+' one.')
                    selected = self._select(now, options, instance, effects, text)
                    if target is not None: target = selected
                elif not self._choice_prompted:
                    self._choice_prompted=True
                    self._speak(effects,'Say left or right.','choose_again')
            elif command.kind=='help':
                self._speak(effects,HELP,'help',1)
            elif command.kind in ('louder','quieter'):
                self._volume(command.kind,effects)
            elif command.kind in ('brief','verbose'):
                self.brief = command.kind=='brief'
                self._speak(effects,'Okay, less talking.' if self.brief else 'Okay, more talking.',
                            'brief_on' if self.brief else 'brief_off',1)
            elif command.kind in ('yes','no'):
                self._speak(effects,f'I heard {command.kind}.',f'{command.kind}_ack',0)
            elif command.kind=='repeat' and self._last_spoken:
                text,key,priority=self._last_spoken
                self._speak(effects,text,key,priority)
            elif command.kind=='recheck' and self.state in ('SEARCHING','GUIDING','TARGET_LOST'):
                effects.append(Effect('force_query'));self._speak(effects,'Checking.','checking',0)

        if self._readback_results(now, results, effects):
            target = None  # Never guide using the just-rejected observation.
        if self._scene_results(now, results, effects, flipped):
            target = None
        center=target_center(target)
        target_ok=(center is not None and not stats.get('stale',False)
                   and target.phrase == (self.phrase or ''))
        # Mouse click targets have no phrase and no network request.
        if center is not None and target.phrase=='' and not self._safety_blocked and self.state not in ('CONTACT','CONFIRMING','DONE','DESCRIBING'):
            if self.phrase != '' or self.state=='IDLE':
                self.phrase='';self.contact.reset();self._many_notice=False
                self._transition(now,'GUIDING',effects,'clicked_target')
                effects.append(Effect('earcon',key='found'))
            target_ok=not stats.get('stale',False)
        hand_ok=(hand is not None and hand.present and hand.fingertip_px is not None
                 and all(math.isfinite(v) for v in hand.fingertip_px))
        state=self.state
        if state=='SEARCHING':
            elapsed=now-(self._search_start if self._search_start is not None else self.entered)
            if target_ok:
                eligible = self._eligible(target)
                if len(eligible)>=2:
                    self._disambiguation=target
                    self._choice_prompted=False
                    self._transition(now,'DISAMBIGUATING',effects,'ambiguous_target')
                    effects.append(Effect('earcon',key='found'))
                    centers=[((i.box_xyxy[0]+i.box_xyxy[2])/2,(i.box_xyxy[1]+i.box_xyxy[3])/2) for i in eligible]
                    horizontal=max(p[0] for p in centers)-min(p[0] for p in centers) >= max(p[1] for p in centers)-min(p[1] for p in centers)
                    count='two' if len(eligible)==2 else 'three'
                    prompt='One on your left, one on your right. Say left or right.' if horizontal else 'One ahead, one closer. Say ahead or closer.'
                    self._speak(effects,f'I see {count} {self.phrase}s. {prompt}','disambiguate')
                else:
                    self._transition(now,'GUIDING',effects,'target_found')
                    effects.append(Effect('earcon',key='found'))
                    if self.brief:
                        self._speak(effects,f'Found {self._name()}.','found_brief')
                    else:
                        self._speak(effects,f'Found {self._name()}, {self._where(hand,center,frame_size,flipped)}.','found')
            elif elapsed+1e-9 >= self._number('give_up_s',20):
                name=self._name();self._clear(now,effects,'search_timeout')
                self._speak(effects,f"I couldn't find {name}.",'not_found')
            elif elapsed+1e-9>=self._number('search_prompt_s',4) and (self._last_prompt is None or now-self._last_prompt+1e-9>=self._number('every_s',6)):
                self._last_prompt=now
                self._speak(effects,f"I can't see {self._name()} yet. Move the phone slowly.",'sweep')
        elif state=='DISAMBIGUATING':
            if target_ok:
                self._disambiguation=target
            if now-self.entered+1e-9>=self._number('disambiguate_s',8):
                options=self._disambiguation
                instance=self._pick(options,None,hand,flipped,nearest=True)
                if instance is not None:
                    selected=self._select(now,options,instance,effects,'Guiding to the nearest one.')
                    if target_ok: target=selected;center=target_center(target)
        elif state in ('GUIDING','HAND_LOST'):

            if not target_ok:
                if self._target_missing is None:self._target_missing=now
                if now-self._target_missing+1e-9>=self._number('target_lost_s',1):
                    self._transition(now,'TARGET_LOST',effects,'target_missing')
                    effects.append(Effect('earcon',key='lost'))
            else:
                self._target_missing=None
                if state=='GUIDING' and not hand_ok:
                    self._transition(now,'HAND_LOST',effects,'hand_missing')
                elif state=='HAND_LOST' and hand_ok:
                    self._transition(now,'GUIDING',effects,'hand_returned')
                elif state=='HAND_LOST' and not self._hand_notice and now-self.entered+1e-9>=self._number('hand_lost_speak_s',1.5):
                    self._hand_notice=True
                    self._speak(effects,'Bring your hand into view.','hand_lost')
        elif state=='TARGET_LOST':
            elapsed=now-(self._target_missing if self._target_missing is not None else self.entered)
            if target_ok:
                self._target_missing=None
                self._transition(now,'GUIDING',effects,'target_returned')
                effects.append(Effect('earcon',key='reacquired'))
            elif elapsed+1e-9>=self._number('target_give_up_s',15):
                name=self._name();self._clear(now,effects,'target_timeout')
                self._speak(effects,f'I lost {name}.','gave_up')
            elif not self._target_notice and elapsed+1e-9>=self._number('target_lost_speak_s',2):
                self._target_notice=True
                self._speak(effects,f'Lost {self._name()}. Hold the phone still.','target_lost')
        elif state == 'DESCRIBING':
            if now-self.entered >= self._scene_deadline:
                self._scene_pending = None
                self._speak(effects, "I couldn't look around just now.", 'describe_error')
                self._transition(now, 'IDLE', effects, 'describe_timeout')
        elif state=='CONFIRMING':
            elapsed=now-self.entered
            if elapsed+1e-9>=float(self.cfg.get('spark',{}).get('deadline_s',6)):
                self._touching(effects)
                self._finish_confirmation(now,effects,'readback_timeout')
            elif elapsed+1e-9>=self._number('readback_wait_s',2.5):
                self._touching(effects)
        elif state=='DONE' and now-self.entered+1e-9>=1:
            self._clear(now,effects,'done')

        if target_ok and self.state in ('GUIDING','HAND_LOST') and stats.get('ambiguous') and not self._many_notice:
            self._many_notice=True;self._speak(effects,'I see more than one.','many',0)
        if stats.get('voice_online',True) is False or stats.get('voice_offline',False):
            if not self._voice_notice:
                self._voice_notice=True;self._speak(effects,'Voice is offline. Use the keys.','voice_off')
        if self.state in ('SEARCHING','GUIDING','TARGET_LOST') and stats.get('consecutive_errors',0)>=2:
            if self._last_network is None or now-self._last_network+1e-9>=self._number('network_speak_every_s',15):
                self._last_network=now;self._speak(effects,'Connection problem. Still trying.','network')

        cue=None
        if self.state=='GUIDING' and target_ok and hand_ok:
            if self.cfg.get('contact',{}).get('enabled',True) and not stats.get('hand_held', False):
                contact = self.contact.update(now, hand, target)
            else:
                # A held pose is useful for brief guidance continuity, but its
                # zero velocity is not evidence that a fresh fingertip is still.
                self.contact.update(now, None, None)
                contact = ContactState()
            self.contact_state=contact
            if contact != self._last_contact:
                effects.append(Effect('log',key='contact',payload={'event':'fire' if contact.fired else 'change',
                    **asdict(contact),'hand_frame_id':hand.frame_id,'target_frame_id':target.frame_id}))
                self._last_contact=contact
            if contact.fired:
                self._begin_confirmation(now,target,effects)
            else:
                if self.params.get('ask_touch_enabled',False) and contact.inside and not self._ask_asked and now>=self._ask_suppressed_until:
                    if self._ask_started is None: self._ask_started=now
                    if now-self._ask_started+1e-9>=self._number('ask_touch_s',1.5):
                        self._ask_asked=True;self._ask_until=now+5
                        self._speak(effects,'Are you touching it?','ask_touch')
                elif not contact.inside:
                    self._ask_started=None
                point=predict_tip(hand.fingertip_px,hand.velocity_px_s,center,float(self.params.get('prediction',self.cfg.get('prediction',.5))))
                dx,dy,distance,cue=map_vector(point,center,*frame_size,flipped=flipped,gain=float(self.cfg.get('audio',{}).get('gain',.2)))
                return self._decision(now,hand,target,cue,effects,(dx,dy,distance))
        else:
            # No dwell can accrue through a missing hand/target or a timer tick.
            self.contact.update(now,None,None)
            self.contact_state=None
        return self._decision(now,hand,target,cue,effects)

    def _decision(self,now,hand,target,cue,effects,vector=(None,None,None)):
        speech=' '.join(e.text for e in effects if e.kind=='speak') or None
        earcon=next((e.key for e in reversed(effects) if e.kind=='earcon'),None)
        result=Decision(self._seq,now,self.state,*vector,cue,speech,earcon,
                        hand.frame_id if hand else None,target.frame_id if target else None)
        effects=[replace(effect,payload=plain(effect.payload)) if effect.payload is not None and effect.key!='brain_in' else effect for effect in effects]
        output=plain({'seq':result.seq,'decision':result,'effects':effects[1:]})
        effects.append(Effect('log',key='brain_out',payload=output))
        self._seq+=1
        return result,effects
