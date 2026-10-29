# Reach

Voice-guided hand-to-object finding for blind and low-vision people, using a phone camera, Meta's SAM 3.1, and stereo sound cues.

> **Status:** in development. Built during the Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026.

Current camera checks use a phone held in one hand while the other hand reaches, on both an Android phone and an iPhone. Body-worn capture has not been validated.

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

Shared defaults live in `config.yaml`. Put machine-specific camera settings in
the ignored `config.local.yaml`; nested values override the defaults. Keep API
keys only in `.env`. Shared messages are defined in `src/reach/types.py`.

MediaPipe Tasks can send usage and performance metrics to Google; see its
[Python documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Target perception integration

`Perception(config, camera, log)` runs scene motion and one segmentation worker.
The camera must already be open and running. It supplies
`wait_newer(last_id, timeout) -> (Frame, image) | None` and
`get_jpeg(frame_id) -> bytes | None`; each frame hash must match that JPEG.
Call `start()`, `set_phrase("wallet")` or `select_point(x, y)`, and feed hand
observations with `on_hand(hand)`. Read `latest()` for guidance and `stats()`
for overlay diagnostics. Call `stop()` before closing `EventLog`.

`latest()` returns no target when motion fails or camera frames are over
500 ms old. A new SAM result can recover a lost target. Use a textured table
or placemat. `motion.method: kcf` selects the appearance-tracker fallback;
its masks are unavailable and it can follow the hand during occlusion.
`force_query()` requests a refresh while retaining the idle, backoff and
one-request guards. Unknown and idle states send no images.

Only `sam-3.1` is accepted by the segmentation client. It makes one streaming
request with no retries and a three-second deadline. Frames and a noun phrase
are sent to Meta's standard API in phrase mode. Click mode makes no API calls.
Default objects are wallet, pen, ball and apple; other nouns in the voice
keywords file have not been measured on these setups.

Set `sam.cache` in the configuration to `off`, `read` or `write`. The default
`auto` uses `write` for file sources and `off` for live sources. `write` reads
existing entries and populates misses; misses send an image. `read` is strictly
local and reports a miss without calling the API. The runner can map its
`--sam-cache` option to this setting. Cache files live under ignored
`runs/sam-cache/` and are keyed by image bytes, phrase, model, confidence cutoff
and dimensions. Run folders can contain sensitive images; keep them local.

The camera, hand, audio and command interfaces still use simulation stubs on
this branch. A real capture source and runner are required for live guidance.
