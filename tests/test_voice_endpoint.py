from types import SimpleNamespace as NS

import numpy as np
import pytest

from reach.voice_endpoint import EnergyEndpoint, capture_end_time


def test_capture_timestamp_mapping_and_invalid_clock():
    assert capture_end_time(10,NS(currentTime=50,inputBufferAdcTime=49.9),1920)==pytest.approx((9.98,True))
    assert capture_end_time(10,NS(currentTime=50,inputBufferAdcTime=0),1920)==(10,False)


def test_endpoint_calibration_subframes_hangover_and_consumed_turn():
    e=EnergyEndpoint()
    silence=bytes(3840)
    for i in range(8):
        e.accept(silence,(i+1)*.08)
    assert e.threshold==.008
    # Only the first 20 ms of an 80 ms block contains speech.
    samples=np.zeros(1920,dtype='<i2')
    samples[:480]=1000
    e.accept(samples.tobytes(),.72)
    assert e.last_voice==pytest.approx(.66) and e.end() is None
    e.accept(silence,.80)
    assert e.end() is None
    e.accept(silence,.88)
    assert e.end()==pytest.approx(.66)
    assert e.end(since=.66) is None
    e.accept(samples.tobytes(),.96,False)
    e.accept(silence,1.2)
    assert e.end() is None
