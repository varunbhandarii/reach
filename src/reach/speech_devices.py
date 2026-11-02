"""Microphone/output checks for stereo guidance with a separate voice input."""

import ctypes
import ctypes.util
import math
import struct
import sys


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


def default_output_route():
    """Read live CoreAudio metadata, without opening or reinitializing a stream."""
    if sys.platform != 'darwin':
        return None
    library = ctypes.util.find_library('CoreAudio')
    if not library:
        raise OSError('CoreAudio is unavailable')
    core = ctypes.CDLL(library)
    getter = core.AudioObjectGetPropertyData
    getter.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Address), ctypes.c_uint32,
                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    getter.restype = ctypes.c_int32

    def get(object_id, selector, value_type):
        fourcc = lambda value: struct.unpack('>I', value.encode('ascii'))[0]
        address = _Address(fourcc(selector), fourcc('glob'), 0)
        value = value_type()
        size = ctypes.c_uint32(ctypes.sizeof(value))
        status = getter(object_id, ctypes.byref(address), 0, None,
                        ctypes.byref(size), ctypes.byref(value))
        if status or size.value != ctypes.sizeof(value):
            raise OSError(status, 'CoreAudio route query failed')
        return value.value

    device = get(1, 'dOut', ctypes.c_uint32)
    if not device:
        return {'device_id': 0, 'rate': 0.0}
    return {'device_id': device, 'rate': get(device, 'nsrt', ctypes.c_double)}


def route_ok(route, start_route):
    # Unknown metadata uses the documented preflight fallback, not a measured pass.
    if route is None:
        return True
    rate = route.get('rate')
    device = route.get('device_id')
    return (isinstance(rate, (int, float)) and not isinstance(rate, bool)
            and math.isfinite(rate) and rate >= 44100 and bool(device)
            and (start_route is None or device == start_route.get('device_id')))


class AudioRouteMonitor:
    """Main-loop polling; never called from the realtime audio callback."""
    def __init__(self, probe=default_output_route, *, platform=None, interval_s=2.0):
        self.probe = probe
        self.platform = platform or sys.platform
        self.interval_s = interval_s
        self.start_route = None
        self.next_poll = None
        self.ok = True
        self.status = None

    def poll(self, now):
        if self.next_poll is not None and now < self.next_poll:
            return None
        self.next_poll = now + self.interval_s
        previous = self.ok
        error = None
        if self.platform != 'darwin':
            route = None
            status = 'not_required'
        else:
            try:
                route = self.probe()
                status = 'known' if route is not None else 'unknown'
            except (OSError, ValueError, TypeError) as exc:
                route, status, error = None, 'unknown', type(exc).__name__
        if self.start_route is None and route and route.get('device_id'):
            self.start_route = dict(route)
        # A query failure cannot silently recover an already established loss.
        self.ok = previous if status == 'unknown' else route_ok(route, self.start_route)
        transition = 'lost' if previous and not self.ok else 'recovered' if not previous and self.ok else None
        self.status = {'rate': route.get('rate') if route else None,
                       'device_id': route.get('device_id') if route else None,
                       'start_device_id': self.start_route.get('device_id') if self.start_route else None,
                       'ok': self.ok, 'status': status, 'transition': transition,
                       'error': error}
        return dict(self.status)


def stereo_warning(output, *, input_name=None):
    if input_name and "airpods" in input_name.casefold() and "airpods" in output.get("name", "").casefold():
        return "AirPods microphone may disable stereo playback even when device metadata stays stereo; use the built-in microphone."
    channels=output.get('max_output_channels',output.get('channels',0))
    rate=output.get('default_samplerate',output.get('samplerate',0))
    if channels<2 or rate<44100:
        return 'Stereo guidance unavailable: choose the built-in microphone and a stereo output at 44.1 kHz or higher.'
    return None


def mac_builtin_input(devices):
    candidates=[(i,d) for i,d in enumerate(devices) if d.get('max_input_channels',0)>0
                and ('macbook' in d.get('name','').casefold() or 'built-in' in d.get('name','').casefold())
                and 'airpods' not in d.get('name','').casefold()]
    if len(candidates)!=1:raise ValueError('Select the Mac built-in microphone explicitly')
    return candidates[0][0]
