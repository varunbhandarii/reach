from pathlib import Path
import numpy as np
import pytest
from reach.masks import encode_rle, decode_rle
from reach.sam_client import parse_instances, SamParseError, SamClient, SamPermanentError, retry_seconds
from reach.types import Frame

FRAME = Frame(0, 0, 0, 640, 480, "hash", "file")
TEXT = (Path(__file__).parent / "fixtures/segmentation.txt").read_text()


def test_recorded_tokens_parse_and_masks_use_half_open_boxes():
    result = parse_instances(TEXT, FRAME)
    assert len(result) == 1
    box = result[0]
    assert box.sam_id == "0" and box.conf > .9
    assert decode_rle(box.mask_rle).shape == (box.box_xyxy[3]-box.box_xyxy[1], box.box_xyxy[2]-box.box_xyxy[0])
    # Enlarge only the source box, keeping the original small raster tokens.
    scaled = parse_instances(TEXT.replace('x2=524', 'x2=599'), FRAME)[0]
    assert decode_rle(scaled.mask_rle).shape == (480, 600)
    assert parse_instances(TEXT, FRAME, .99) == ()
    assert parse_instances('', FRAME) == ()


def test_invalid_output_and_wrong_frame_rejected():
    for value in ('garbage', TEXT.replace('w=640', 'w=600'), TEXT.split('<|mask;')[0]):
        with pytest.raises(SamParseError):
            parse_instances(value, FRAME)


@pytest.mark.parametrize('shape', [(1, 1), (23, 17), (0, 0)])
def test_rle_round_trip(shape):
    for mask in (np.zeros(shape, np.uint8), np.ones(shape, np.uint8), np.random.default_rng(4).integers(0, 2, shape)):
        assert np.array_equal(mask, decode_rle(encode_rle(mask)))


def test_client_guards_and_backoff():
    with pytest.raises(ValueError):
        SamClient({'sam': {'model': 'sam-3.1-contributor'}})
    with pytest.raises(SamPermanentError):
        SamClient().segment(b'', 'wallet', FRAME)
    assert retry_seconds('2.5') == 2.5
    assert retry_seconds('bad') == 1


def test_stream_transport_status_and_completion(monkeypatch):
    import io
    import json
    from reach.sam_client import SamRateLimited, SamNetworkError, SamTimeout
    payload = b''.join(('data: '+json.dumps(e)+'\n\n').encode() for e in [
        {'type': 'response.output_text.delta', 'delta': TEXT}, {'type': 'response.completed'}])
    class Response(io.BytesIO):
        status = 200
        def getheader(self, name):
            return '2'
    class Connection:
        sock = None
        def __init__(self, *args, **kwargs): pass
        def request(self, *args):
            self.body = args[2]
            assert json.loads(self.body)['store'] is False
        def getresponse(self): return response
        def close(self): pass
    monkeypatch.setattr('reach.sam_client.http.client.HTTPSConnection', Connection)
    # Mask decoding has dedicated tests above; transport/status coverage must
    # not race the request watchdog while decoding a large fixture on CI.
    from reach.types import Instance
    parsed=[]
    def parse(text, frame, conf_min):
        parsed.append((text, frame, conf_min))
        return (Instance('0',(10,10,20,20),.9,None),)
    monkeypatch.setattr('reach.sam_client.parse_instances',parse)
    response = Response(payload)
    assert SamClient(api_key='test-placeholder').segment(b'jpeg', 'wallet', FRAME)[0].sam_id == '0'
    assert parsed == [(TEXT,FRAME,.5)]
    response = Response(b'')
    response.status = 429
    with pytest.raises(SamRateLimited) as err:
        SamClient(api_key='test-placeholder').segment(b'jpeg', 'wallet', FRAME)
    assert err.value.retry_after == 2
    response.status = 400
    with pytest.raises(SamPermanentError):
        SamClient(api_key='test-placeholder').segment(b'jpeg', 'wallet', FRAME)
    response.status = 200
    with pytest.raises(SamNetworkError):
        SamClient(api_key='test-placeholder').segment(b'jpeg', 'wallet', FRAME)
    def slow():
        import time
        time.sleep(.04)
        return Response(payload)
    Connection.getresponse = lambda self: slow()
    with pytest.raises(SamTimeout):
        SamClient({'sam': {'deadline_s': .02}}, api_key='test-placeholder').segment(b'jpeg', 'wallet', FRAME)
