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
the complete turn. Reports also separate a local speech-end estimate using
capture timestamps, calibrated energy in 20 ms subframes and a 200 ms hangover.
This estimate is not phoneme-aligned ground truth; turns with audio drops,
overflows or invalid capture timestamps have no valid local latency.
Tap-to-talk measures the second tap to the final transcript; these measurements
have different starting points.
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
`reach replay-contact RUN_DIR --frames` reconstructs contact from saved reducer
inputs and masks when available, and labels older box-only logs explicitly.
It runs without network or device access; logged geometry cannot prove touch.

## Touched-object read-back and recording

With the confirming reducer installed, contact starts a single asynchronous
standard `muse-spark-1.3` request while the chime plays immediately. The marked
crop includes the target and hand. Requests explicitly set `store: false` and
contributor model IDs are refused. Standard mode does not use content for model
training; disabling response storage does not establish a retention guarantee.
See Meta's [Responses API](https://dev.meta.ai/docs/protocols/responses) and
[pricing modes](https://dev.meta.ai/docs/pricing-rate-limits).

`--readback off` disables image read-back. `--spark-cache read` uses saved local
answers only, including on cache misses. `auto` reads and writes a local cache
for file sources and disables caching for live sources. The default effort is
`minimal`, with image detail `low`; API and audio timings are reported separately.
Use `reach readback-probe crop.jpg --phrase wallet --count 5` to measure a saved
crop explicitly. This command uploads the crop and prints valid answers and
errors, rather than opening a device.

`--record` saves `overlay.mp4` at 30 fps and the exact rendered stereo output
mix as `output.wav`, with their monotonic start times in the log. Keep both
files to align them in an editor. `--record-format avi` uses MJPEG if the editor
cannot import MP4. Audio recording uses a bounded writer ring; overruns are
logged and silence fills lost blocks to preserve the timeline. Recordings and
read-back caches stay local under ignored `runs/`. Recording requires audio
output; it does not save raw microphone audio.

The on-screen cost total is an estimate from configured prices, images sent,
streamed voice seconds and returned Spark token usage. It is not an invoice or
a statement of available credits.

Controller confirmation, spatial selection and deterministic offline replay are
documented in [the integration reference](docs/confirmation.md).
`reach replay RUN_DIR` verifies recorded inputs/outputs without a camera or API;
`--strip OUTPUT_DIR` writes a fresh events-only copy with canonical command text.

When a clean pre-reach target crop is available, read-back compares that reference
with the contact crop. The result answers whether the same physical object was
touched, so a different name such as "stylus" does not by itself reject a pen.
Without a clean reference it uses the single-image check and reports that mode.
`reach readback-probe contact.jpg --reference reference.jpg --phrase pen --count 3`
explicitly uploads both saved images. Cache identities include both image hashes.

## Scene listing and natural requests

The scene client lists up to six visible objects and their image positions. Its
worker runs independently of read-back, uses a six-second deadline and keeps a
local cache for file sources. The default image size is 512 pixels on the long
side with low detail; `scene.image_long_side` and `scene.detail` are configurable.
Press **D** or ask "what's on the table?",
then request one of the listed objects by name.

When the local grammar cannot understand a request, a bounded text-only fallback
can interpret everyday phrasing or another language. It sends text only when
there is a request cue or non-ASCII lettering, uses a three-second deadline, and
never retries automatically. Set `parse.enabled: false` to disable it. Parsed
commands retain their source and language; a late reply cannot undo a voice stop.
Windows speech replies stay English. Scene and fallback requests use standard
Spark with `store: false`, as described in Meta's
[Responses API](https://dev.meta.ai/docs/protocols/responses).

Sharp and hot object requests receive a spoken refusal before any grounding
request. Scene lists mention a recognized hazard only as a warning. Refusals
apply equally to local commands and parsed natural requests. This tool is for
non-hazardous tabletop objects; object names returned by a model can be wrong.

## Mac stereo and Hindi speech

Use the MacBook microphone for input and select AirPods as the system output.
Reach reads the live CoreAudio output route every two seconds. A low sample
rate or a change away from the starting output mutes guidance, announces the
loss and blocks starting a timed trial. Reconnecting the original stereo
output restores guidance and announces recovery. Unknown route metadata is
labeled explicitly; it is not a stereo measurement. Selecting an AirPods
microphone is refused before opening the input stream.

On macOS, English and Hindi request bias and the installed Lekha voice are
configured by default. `tts.voices` selects a voice per language. Hindi commands
use their spoken object names in checked-in Hindi templates; SAM still receives
an English target. Fixed phrases for both languages are rendered before the
microphone starts. Windows continues to speak English. The phrase tables in
`config/phrases/` can be reviewed or edited without making translation calls.

Keep a round object supported on a folded cloth so it cannot roll, and touch
with a visible index fingertip before lifting it. Briefly held hand poses may
maintain guidance, but cannot accumulate contact dwell time.

## Sound display and event timelines

The display shows left/right pan, forward/back pitch, a blinking pulse and an
ON LINE marker alongside larger speech captions. `--overlay-scale 2` draws text
and shapes after scaling the camera image, producing a sharp 1280×960 canvas
from a 640×480 source. Mouse selection still uses source-image coordinates.
`--record` saves that scaled canvas and the rendered stereo mix locally.

`reach report RUN_DIR --html timeline.html` exports a self-contained HTML/SVG
timeline with state, distance, pan, pitch, commands, speech, model activity,
contact and trial lanes. Hover over bars or open the event table for details.
The report labels latency origins and omits raw microphone transcripts. Stripped
events-only runs work without devices, image files or API access; missing
measurements remain visibly absent.
