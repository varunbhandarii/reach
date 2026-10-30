from dataclasses import replace
from hashlib import sha256
import threading
import time
import cv2
import numpy as np
from reach.perception import Perception
from reach.scheduler import QueryScheduler
from reach.types import Frame, Instance


class Camera:
    def __init__(self):
        self.image = cv2.GaussianBlur(np.random.default_rng(6).integers(0, 256, (480, 640, 3), np.uint8), (5, 5), 0)
        self.jpeg = cv2.imencode('.jpg', self.image)[1].tobytes()
    def get_jpeg(self, frame_id):
        return self.jpeg
    def frame(self, i):
        return Frame(i, time.perf_counter(), 0, 640, 480, sha256(self.jpeg).hexdigest(), 'file')


def test_scheduler_silence_rates_backoff_and_inflight():
    s = QueryScheduler()
    for state in ('IDLE', 'UNKNOWN', 'STOPPED', 'CONTACT'):
        assert not s.should_query(state, False, 0)
    assert s.should_query('SEARCHING', False, 0)
    assert not s.should_query('SEARCHING', False, .4)
    assert s.should_query('SEARCHING', False, .5)
    assert not s.should_query('GUIDING', True, 1)
    assert s.should_query('GUIDING', True, 2.5)
    assert s.should_query('GUIDING', True, 3, near=True)
    assert not s.should_query('SEARCHING', False, 4, in_flight=True)
    s.backoff(4, 3)
    s.force()
    assert not s.should_query('SEARCHING', False, 6)
    assert s.should_query('SEARCHING', False, 7)


def test_click_and_idle_do_not_send_requests():
    c = Camera()
    p = Perception({}, c, None)
    p.process(c.frame(0), c.image)
    assert p.stats()['images_sent'] == 0
    p.select_point(300, 250)
    p.process(c.frame(1), c.image)
    assert p.latest().instances[0].box_xyxy == (260, 210, 340, 290)
    assert p.stats()['images_sent'] == 0
    p.set_phrase(None)
    assert p.latest() is None
    p.process(c.frame(2), c.image)
    assert p.latest() is None


def test_one_inflight_and_changed_phrase_drops_old_response():
    c = Camera()
    p = Perception({}, c, None)
    p.set_phrase('wallet')
    p.process(c.frame(0), c.image)
    job = p._jobs.get_nowait()
    for i in range(1, 4):
        p.process(c.frame(i), c.image)
    assert p._req_id == 1
    p.set_phrase('apple')
    p._results.put((job, (Instance('1', (100, 200, 150, 250), .9, None),), None, 100))
    p.process(c.frame(4), c.image)
    assert p.latest() is None
    assert p.stats()['responses_dropped'] == 1
    assert p._req_id == 2


def test_selection_preserves_overlap_and_marks_ambiguity():
    c = Camera()
    p = Perception({}, c, None)
    p.set_phrase('apple')
    p.process(c.frame(0), c.image)
    job = p._jobs.get_nowait()
    instances = (Instance('a', (100, 200, 150, 250), .95, None), Instance('b', (300, 200, 350, 250), .9, None))
    p._results.put((job, instances, None, 100))
    p.process(c.frame(1), c.image)
    assert p.latest().instances[0].sam_id == 'a'
    assert p.stats()['ambiguous']
    # Refresh confidence favours b, but overlap keeps a.
    job = {**job, 'frame': c.frame(1), 'submitted': time.perf_counter()}
    p._inflight = job
    p._results.put((job, (replace(instances[0], conf=.8), replace(instances[1], conf=.99)), None, 100))
    p.process(c.frame(2), c.image)
    assert p.latest().instances[0].sam_id == 'a'


def test_deadline_drop_even_with_valid_history():
    c = Camera()
    p = Perception({}, c, None)
    p.set_phrase('wallet')
    p.process(c.frame(0), c.image)
    job = p._jobs.get_nowait()
    job['submitted'] -= 4
    p._results.put((job, (Instance('1', (100, 200, 150, 250), .9, None),), None, 4000))
    p.process(c.frame(1), c.image)
    assert p.latest() is None and p.stats()['responses_dropped'] == 1


def test_latest_silences_disconnected_camera():
    c = Camera()
    p = Perception({}, c, None, clock=lambda: 100)
    p.select_point(300, 250)
    f = replace(c.frame(0), t_mono=99)
    p.process(f, c.image)
    assert p.latest() is None


def test_real_worker_never_overlaps_calls_and_stops():
    c = Camera()
    released = threading.Event()
    entered = threading.Event()
    calls = []
    def segment(*args):
        calls.append(args)
        entered.set()
        assert released.wait(2)
        return (Instance('1', (100, 200, 150, 250), .9, None),)
    p = Perception({}, c, None, segmenter=segment)
    p.set_phrase('wallet')
    worker = threading.Thread(target=p._sam_worker)
    worker.start()
    try:
        p.process(c.frame(0), c.image)
        assert entered.wait(2)
        for i in range(1, 8):
            p.process(c.frame(i), c.image)
        assert len(calls) == 1
        released.set()
        deadline = time.perf_counter()+2
        while p._results.empty() and time.perf_counter() < deadline:
            time.sleep(.005)
        p.process(c.frame(8), c.image)
        assert p.latest() is not None
    finally:
        released.set()
        p._stop.set()
        worker.join(2)
    assert not worker.is_alive()


class CapturedLog:
    def __init__(self):
        self.events = []
    def emit(self, kind, data):
        self.events.append((kind, data))


def test_stop_records_inflight_response_before_log_close():
    c, log = Camera(), CapturedLog()
    entered, release = threading.Event(), threading.Event()
    def segment(*args):
        entered.set()
        assert release.wait(2)
        return (Instance('1', (100, 200, 150, 250), .9, None),)
    p = Perception({}, c, log, segmenter=segment)
    p.set_phrase('wallet')
    p.process(c.frame(0), c.image)
    p._worker_thread = threading.Thread(target=p._sam_worker)
    p._worker_thread.start()
    assert entered.wait(2)
    timer = threading.Timer(.02, release.set)
    timer.start()
    try:
        p.stop()
    finally:
        release.set()
        timer.join()
        p._worker_thread.join(2)
    requests = [data for kind, data in log.events if kind == 'sam_req']
    responses = [data for kind, data in log.events if kind == 'sam_resp']
    assert len(requests) == len(responses) == 1
    assert responses[0]['req_id'] == requests[0]['req_id']
    assert responses[0]['status'] == 'dropped' and responses[0]['reason'] == 'shutdown'
    assert p.latest() is None
    assert not p.stats()['in_flight'] and p.stats()['state'] == 'STOPPED'
    p.stop()
    assert len([e for e in log.events if e[0] == 'sam_resp']) == 1


def test_stop_cancels_queued_work_without_claiming_an_api_request():
    c, log = Camera(), CapturedLog()
    p = Perception({}, c, log)
    p.set_phrase('wallet')
    p.process(c.frame(0), c.image)
    p.stop()
    assert p.stats()['images_sent'] == 0
    assert not p.stats()['in_flight'] and p._jobs.empty()
    assert [kind for kind, _ in log.events if kind in ('sam_req', 'sam_resp')] == []
    assert [data['reason'] for kind, data in log.events if kind == 'sam_cancel'] == ['shutdown_before_send']
