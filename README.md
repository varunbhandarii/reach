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

Capture, hand tracking, stereo guidance, target perception and the overlay are
implemented. Voice commands, spoken read-back and automatic contact detection
are still under development. Touch outcomes in timed trials are entered by the
operator.

Save the [official hand model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
as `models/hand_landmarker.task`, or set `hand.model_path` in your local config.
Choose a stereo output with `uv run reach devices`. Use wired headphones and low
system volume for the direction check.

```sh
uv run reach run --mount handheld
uv run reach run --mount handheld --source path/to/clip.mp4 --mute --sam-cache read
uv run reach earcons
```

Click a point to guide toward it without API requests. Keys 1–4 choose wallet,
pen, ball and apple and send images to the configured Meta segmentation API;
`c` clears, `r` rechecks, `m` toggles sound and `q` quits. File runs use the cache
automatically; `--sam-cache read` prevents network requests on cache misses.

`--predict 0` disables velocity prediction; `--predict 0.5` enables it. Hand
inference defaults to two hands. `--hand-width 480 --max-hands 1` is an explicit
performance fallback for a single reaching hand; it disables two-hand selection.
The overlay detects an inverted wrist orientation; use `--flip off` or `on`
after checking cue directions if automatic flipping is unsuitable.

Run `--duration 600` for an endurance recording. Optional `--trials 5
--trial-mode click` or `phrase` provides timed trials: Space starts once a target
is ready, `t` records a reported touch, `f` records failure, and `n` prepares the
next trial. The default time limit is 45 seconds. These keys record operator
reports; fingertip proximity never counts as a touch automatically.

Each run saves a manifest, events, JPEG frames and a summary under ignored
`runs/`. `uv run reach report path/to/run` summarizes timing, audio underflows and
trial paths. `--camera-latency-ms` adds an earlier measured camera median to a
component estimate; this is not a measured end-to-end latency distribution.

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

Live camera capture, hand tracking, stereo guidance, overlay and run logging
are implemented. Voice-command and speech-output interfaces still contain
simulation stubs.

Late segmentation results also receive a conservative appearance check using
`camera.get(frame_id)` and the stored source mask. A large color disagreement
in a clear target region clears guidance and requests a fresh frame. Visible
hand occlusion exempts this check. `motion.appearance_guard` controls it; the
check detects gross changes and does not establish object identity. Repeated
selection of the same phrase refreshes the search without discarding a pending
result or a current lock.

The deterministic command/session/contact components and local speech engine
are described in [session integration](docs/session-integration.md).
`reach replay-contact RUN_DIR --frames` inspects saved box/contact geometry
without network or device access. Live voice and mixed speech need the runner
integration to consume these interfaces.
