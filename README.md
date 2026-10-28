# Reach

Voice-guided hand-to-object finding for blind and low-vision people. Reach aims
to find named tabletop objects with Meta's SAM 3.1 and use stereo sound cues to
guide the reaching hand toward them.

> **Status:** in development. Built during the Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026.

Reach is a prototype for locating non-hazardous objects on a tabletop within arm's reach. It is not a medical device and not a mobility aid.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Works on Windows and macOS.

```
uv python install 3.12
uv sync
uv run pytest
```

Copy `.env.example` to `.env` and add your own Meta Model API key. Never commit `.env`.

Camera, hand, audio, voice and speech interfaces currently contain simulation
stubs. They return synthetic observations or retain requests without opening
hardware or making API calls. They do not provide live guidance yet.

Current camera checks use a phone held in one hand while the other hand reaches.
Body-worn capture has not been validated. A phone supported steadily on the desk
can be explored separately; camera placement must be reported with its results.

Shared defaults live in `config.yaml`. Put machine-specific camera settings in
the ignored `config.local.yaml`; nested values override the defaults. Keep API
keys only in `.env`. Shared messages are defined in `src/reach/types.py`.

MediaPipe Tasks can send usage and performance metrics to Google; see its
[Python documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

## License

Apache-2.0. See [LICENSE](LICENSE).
