"""Deterministic session reducer; all time and observations come from the caller.

The hand-thread owner calls step every hand frame and on no-frame ticks.
Pass hand=None when its observation expires. perception_stats may include
voice_online=False (or voice_offline=True) for the once-per-run voice notice.
Effects use the existing log type/data envelope and perform no I/O here.
"""

from dataclasses import asdict
import math
from .contact import ContactRule, ContactState
from .controller import target_center, predict_tip, map_vector
from .types import Decision, Effect


class Brain:
    def __init__(self, cfg=None, objects=('wallet','pen','ball','apple')):
        self.cfg = cfg or {}
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

    def _number(self, key, default):
        return float(self.params.get(key, default))

    def _speak(self, effects, text, key, priority=1):
        self._last_spoken = (text,key,priority)
        effects.append(Effect('speak', text, key, priority))

    def _name(self):
        return f'the {self.phrase}' if self.phrase else 'the selected point'

    def _transition(self, now, state, effects, reason, *, force=False):
        old = self.state
        if old == state and not force:
            return
        rate = lambda s: 'IDLE' if s in ('IDLE','CONTACT','DONE') else s
        self.state, self.entered = state, now
        effects.append(Effect('log', key='state', payload={'from':old,'to':state,'reason':reason,'phrase':self.phrase}))
        if rate(old) != rate(state) or force:
            effects.append(Effect('perception_state', key=rate(state)))
        if state == 'HAND_LOST': self._hand_notice = False
        if state == 'TARGET_LOST': self._target_notice = False

    def _clear(self, now, effects, reason):
        self._transition(now,'IDLE',effects,reason)
        self.phrase = None
        self._target_missing = None
        self._search_start = None
        self.contact.reset()
        effects.append(Effect('set_phrase', text=None))

    def _find(self, now, phrase, effects):
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
        if depth and side: return f'{depth} on your {side}'
        if side: return f'to your {side}'
        if depth=='back': return 'back toward you'
        return 'straight ahead'

    def step(self, now, *, hand=None, target=None, perception_stats=None, commands=(), speaking=False,
             frame_size=(640,480), flipped=False):
        if not math.isfinite(now) or (self._last_now is not None and now < self._last_now):
            raise ValueError('session timestamps must be finite and ordered')
        if min(frame_size)<=0:
            raise ValueError('frame dimensions must be positive')
        self._last_now=now
        stats=perception_stats or {}
        effects=[]
        commands=tuple(commands)
        # Stop dominates a queued batch so a cancelled search never reaches I/O.
        stopping=next((c for c in reversed(commands) if c.kind in ('stop','cancel','start_over')),None)
        if stopping:
            effects.append(Effect('cancel_speech'))
            veto = stopping.kind=='cancel' and self.state=='SEARCHING' and now-self.entered <= self._number('veto_s',1.2)+1e-9
            self._clear(now,effects,stopping.kind)
            text,key = ('Ready.','ready') if stopping.kind=='start_over' else ('Cancelled.','cancelled') if veto else ('Stopped.','stopped')
            self._speak(effects,text,key,2)
            return self._decision(now,hand,target,None,effects)
        for command in commands:
            if command.kind=='find' and command.phrase:
                phrase=command.phrase.strip()
                if len(phrase)<=120 and not any(c in phrase for c in ',;\n'):
                    self._find(now,phrase,effects)
            elif command.kind=='no' and self.state=='SEARCHING' and now-self.entered<=self._number('veto_s',1.2)+1e-9:
                effects.append(Effect('cancel_speech'));self._clear(now,effects,'veto')
                self._speak(effects,'Cancelled.','cancelled',2)
            elif command.kind=='repeat' and self._last_spoken:
                text,key,priority=self._last_spoken
                self._speak(effects,text,key,priority)
            elif command.kind=='recheck' and self.state in ('SEARCHING','GUIDING','TARGET_LOST'):
                effects.append(Effect('force_query'));self._speak(effects,'Checking.','checking',0)

        center=target_center(target)
        target_ok=(center is not None and not stats.get('stale',False)
                   and target.phrase == (self.phrase or ''))
        # Mouse click targets have no phrase and no network request.
        if center is not None and target.phrase=='' and self.state not in ('CONTACT','DONE'):
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
                self._transition(now,'GUIDING',effects,'target_found')
                effects.append(Effect('earcon',key='found'))
                self._speak(effects,f'Found {self._name()}, {self._where(hand,center,frame_size,flipped)}.','found')
            elif elapsed+1e-9 >= self._number('give_up_s',20):
                name=self._name();self._clear(now,effects,'search_timeout')
                self._speak(effects,f"I couldn't find {name}.",'not_found')
            elif elapsed+1e-9>=self._number('search_prompt_s',4) and (self._last_prompt is None or now-self._last_prompt+1e-9>=self._number('every_s',6)):
                self._last_prompt=now
                self._speak(effects,f"I can't see {self._name()} yet. Move the phone slowly.",'sweep')
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
        elif state=='CONTACT' and (not speaking or now-self.entered+1e-9>=2):
            self._transition(now,'DONE',effects,'contact_announced')
            effects.append(Effect('set_phrase',text=None))
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
            contact=(self.contact.update(now,hand,target) if self.cfg.get('contact',{}).get('enabled',True)
                     else ContactState())
            self.contact_state=contact
            if contact != self._last_contact:
                effects.append(Effect('log',key='contact',payload={'event':'fire' if contact.fired else 'change',
                    **asdict(contact),'hand_frame_id':hand.frame_id,'target_frame_id':target.frame_id}))
                self._last_contact=contact
            if contact.fired:
                self._transition(now,'CONTACT',effects,'contact')
                effects.append(Effect('earcon',key='contact'))
                self._speak(effects,f"You're touching {self._name()}.",'contact',2)
            else:
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
        self._seq+=1
        return result,effects
