"""Explicit, self-reported timed trials; touching is never inferred from distance."""

import random
import time


class Trials:
    def __init__(self, count, mode, objects, log, *, seed=1, limit=45, clock=time.perf_counter):
        if count < 1 or not objects or limit <= 0:
            raise ValueError("Trials need a positive count, time limit and nonempty objects")
        self.count, self.mode, self.objects = count, mode, objects
        self.log, self.limit, self.clock = log, limit, clock
        self.rng = random.Random(seed)
        self.index = 0
        self.started = None
        self.phrase = None
        self.prepared = False
        self.finished = False
        self.results = []
        self.contact_fired_s = None

    def prepare(self, perception, frame):
        if self.finished or self.prepared:
            return
        self.log.trial = f"t{self.index+1:04d}"
        self.contact_fired_s = None
        if self.mode in ("phrase", "voice"):
            self.phrase = self.rng.choice(self.objects)
            if self.mode == 'phrase':
                perception.set_phrase(self.phrase)
            elif hasattr(perception, 'announce_trial'):
                perception.announce_trial(self.index+1, self.phrase)
        else:
            self.phrase = "clicked point"
            perception.select_point(frame.width/2, frame.height/2)
        self.prepared = True
        self.log.emit("trial_prepare", {"index": self.index+1, "phrase": self.phrase, "mode": self.mode})
        print(f"Trial {self.index+1}/{self.count}: {self.phrase}. "
              f"{'Say the find command. ' if self.mode == 'voice' else ''}"
              'Wait for target; SPACE starts, then cover eyes before reaching.', flush=True)

    def start(self, ready):
        if self.prepared and self.started is None and ready:
            self.started = self.clock()
            self.log.emit("trial_start", {"index": self.index+1, "phrase": self.phrase, "limit_s": self.limit})
            return True
        return False

    def finish(self, success, reason, perception):
        if self.started is None:
            return
        elapsed = self.clock()-self.started
        result = {"index": self.index+1, "phrase": self.phrase,
                  "success": bool(success and elapsed <= self.limit), "elapsed_s": elapsed,
                  "reason": reason, "outcome_source": "manual_touch_key" if success else reason,
                  "contact_fired_s": self.contact_fired_s}
        self.log.emit("trial_end", result)
        self.results.append(result)
        perception.set_phrase(None)
        self.index += 1
        self.started = None
        self.prepared = False
        self.finished = self.index >= self.count
        print(f"Trial result: {result}", flush=True)

    def tick(self, perception):
        if self.started is not None and self.clock()-self.started >= self.limit:
            self.finish(False, "timeout", perception)

    def note_contact(self, now):
        if self.started is not None and self.contact_fired_s is None:
            self.contact_fired_s = max(0.0, now-self.started)

    def lines(self):
        if self.finished:
            return [f"Trials finished: {sum(x['success'] for x in self.results)}/{self.count}; q quits"]
        if self.started is not None:
            return [f"Trial {self.index+1}: {self.clock()-self.started:.1f}/{self.limit:.0f}s | t touched | f failed"]
        return [f"Trial {self.index+1}: {self.phrase or 'preparing'} | SPACE starts when ready", "After each result: N next trial (Y/N answer outside trials)"]
