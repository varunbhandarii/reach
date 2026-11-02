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
implemented, along with microphone commands, local speech mixing, the session
reducer and the contact heuristic. Live recognition and contact accuracy still
need hardware validation. The independent `reach speak` command checks local
speech without opening the microphone or camera.

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
`c` cancels, `r` rechecks, `y`/`n` answer, `m` toggles sound and `q` quits.
`v` starts tap-to-talk; tap it again after speaking to request the final transcript.
File runs use the cache
automatically; `--sam-cache read` prevents network requests on cache misses.

`--predict 0` disables velocity prediction; `--predict 0.5` enables it. Hand
inference defaults to two hands. `--hand-width 480 --max-hands 1` is an explicit
performance fallback for a single reaching hand; it disables two-hand selection.
The overlay detects an inverted wrist orientation; use `--flip off` or `on`
after checking cue directions if automatic flipping is unsuitable.

Live runs open the configured microphone and stream audio to Meta for commands.
`--no-voice` disables this; file and muted runs disable it unless `--voice` is
explicit. `--voice-input "Microphone Array"` selects the laptop input on Windows.
Use headphones to keep synthesized speech out of the microphone. `--speech off`
keeps captions without spoken output; `--contact off` disables automatic contact.

Run `--duration 600 --speech-every 5` for an endurance run with periodic local
speech. Optional `--trials 5 --trial-mode click`, `phrase` or `voice` provides
timed trials. Voice mode announces the scripted target and waits for its find
command. Space starts once a target is ready; for an eyes-covered trial press
Space with eyes open, then cover eyes before reaching. Stop reaching before
uncovering to press `t` for a reported touch or `f` for failure. `n` prepares the
next trial. The default time limit is 45 seconds. Automatic contact timestamps
and operator reports are recorded separately; a chime does not establish touch.

An interrupted scripted run can continue in a new folder with
`--resume-trials path/to/old/run --trials ORIGINAL_TOTAL`. Keep its original
objects, trial mode, seed, time limit, mount, prediction and camera source.
Completed trial outcomes, including failures, are retained; the interrupted
trial starts again with a fresh timer and its original target. The old files are
read only and fingerprinted. Reports label the recovered sequence and preserve
the original interruption separately; current-session timing and errors remain
separate from the earlier process.

Each run saves a manifest, events, JPEG frames and a summary under ignored
`runs/`. Logs can contain microphone transcripts as well as images; keep runs
local and do not commit them. Raw microphone audio is not saved.
`uv run reach report path/to/run` summarizes voice and speech timing, reconnects,
state durations, contact firings, audio underflows and trial paths.
Voice latency measures receipt of the server's speech-end event to receipt of
the complete turn. Tap-to-talk measures the second tap to the final transcript;
the two measurements have different starting points.
`--camera-latency-ms` adds an earlier measured camera median to a
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

## Local speech

`uv run reach speak "Found the wallet." --count 5` synthesizes speech into the
configured stereo guidance stream without opening the camera or microphone.
Windows uses SAPI in memory; macOS expects the shared `say_engine` adapter.
Unavailable engines retain captions and report a fallback. Linux uses silent
samples for device-free tests. These fallbacks do not validate audible speech.
Speech is cached and trimmed before playback, and ducks guidance by 16 dB.
The reported first-audio time ends at the output callback; it excludes the
device's acoustic playback delay.

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


Controller confirmation, spatial selection and deterministic offline replay are documented in [the integration reference](docs/confirmation.md). `reach replay RUN_DIR` verifies recorded inputs/outputs without a camera or API; `--strip OUTPUT_DIR` writes a fresh events-only copy with canonical command text. The live read-back worker and recording adapter are separate integration components.
