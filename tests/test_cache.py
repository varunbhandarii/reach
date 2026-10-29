import json
import threading
import time
from reach.sam_cache import SamCache
from reach.perception import Perception
from reach.types import Frame, Instance

F = Frame(2, 0, 0, 640, 480, 'x', 'file')
I = (Instance('1', (1, 2, 8, 9), .9, None),)


def test_cache_identity_modes_and_corruption(tmp_path):
    cfg = {'camera': {'source': 'file'}, 'sam': {'cache_dir': str(tmp_path)}}
    cache = SamCache(cfg)
    assert cache.mode == 'write'
    assert cache.read(b'jpeg', 'wallet', F) is None
    cache.write(b'jpeg', 'wallet', F, I)
    assert cache.read(b'jpeg', 'wallet', F) == I
    assert cache.read(b'other', 'wallet', F) is None
    assert cache.read(b'jpeg', 'apple', F) is None
    assert SamCache({'sam': {'cache_dir': str(tmp_path), 'model': 'other', 'cache': 'read'}}).read(b'jpeg', 'wallet', F) is None
    next(tmp_path.glob('*.json')).write_text('invalid')
    assert cache.read(b'jpeg', 'wallet', F) is None


def test_read_mode_miss_never_calls_network(tmp_path):
    called = []
    p = Perception({'sam': {'cache': 'read', 'cache_dir': str(tmp_path)}}, None, None,
                   segmenter=lambda *args: called.append(args))
    job = {'id': 0, 'frame': F, 'jpeg': b'jpeg', 'phrase': 'wallet', 'generation': 0, 'submitted': time.perf_counter()}
    p._jobs.put(job)
    worker = threading.Thread(target=p._sam_worker)
    worker.start()
    try:
        _, result, error, _ = p._results.get(timeout=2)
        assert error.kind == 'cache_miss' and called == []
    finally:
        p._stop.set()
        worker.join(1)


def test_cached_empty_result_skips_network(tmp_path):
    cfg = {'camera': {'source': 'file'}, 'sam': {'cache_dir': str(tmp_path)}}
    cache = SamCache(cfg)
    cache.write(b'jpeg', 'wallet', F, ())
    assert cache.read(b'jpeg', 'wallet', F) == ()
