"""Microphone/output checks for stereo guidance with a separate voice input."""


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
