# Session integration

The session core is device-free. The hand-thread owner calls `Brain.step` for
each observation and on a 50 ms timer when capture is silent. Pass `hand=None`
after the hand observation expires; repeated frame ids cannot accumulate
contact dwell. All timestamps use the caller's monotonic clock.

```python
brain = Brain(cfg, cfg['objects'])
decision, effects = brain.step(now, hand=hand, target=perception.latest(),
    perception_stats={**perception.stats(), 'voice_online': voice.stats()['connected']},
    commands=drained_commands, speaking=speaker.busy,
    frame_size=(frame.width, frame.height), flipped=flipped)
audio.set(decision.cue)
for effect in effects:
    if effect.kind == 'log':
        log.emit(effect.key, effect.payload)
    # Dispatch the remaining kinds using the table below.
```

| Effect | Dispatch |
|---|---|
| speak | `speaker.say(effect.text, key=effect.key, priority=effect.priority)` |
| cancel_speech | `speaker.cancel()` |
| earcon | `audio.earcon(effect.key)` |
| set_phrase | `perception.set_phrase(effect.text)`; None clears |
| force_query | `perception.force_query()` |
| perception_state | `perception.set_state(effect.key)` |
| log | `log.emit(effect.key, effect.payload)`; payload is flat event data |
| choose_instance | `perception.choose(effect.payload["sam_id"])` |
| exclude_instance | `perception.exclude(effect.payload["frame_id"], effect.payload["box_xyxy"])` |
| readback | Pass the source-frame crop and request identity to the read-back worker |

Keys, scripted trials and voice post `Command` values through the same queue.
Use `grammar.parse(text, objects, vocabulary, t_mono=now, turn_id=turn,
source='voice')`; it reads no clock. Ignore `unknown` commands, but log their
arrival as appropriate. The runner marks a find phrase outside `objects` as
`untested_noun`. Mouse targets support the legacy empty-phrase path; the integrated runner labels its queued click command and observation as `clicked point`.

The brain owns session states. Perception preserves externally supplied
HAND_LOST/TARGET_LOST states across result refreshes, and IDLE stops queries.
Its diagnostics expose `instances_last`, `consecutive_errors`, and
`last_ok_age_ms` (successful response age, distinct from target/mask age).
Target-loss timers run from the first absent observation. Target speech is at
2 s, give-up at 15 s; hand speech is 1.5 s after entering HAND_LOST. Contact can
fire only in GUIDING when `contact.enabled` is true. `brain.prediction` carries the CLI prediction setting. A duplicate find is a recheck, not a new contact session.

`ContactRule` uses a 150 ms relative-vector history, a per-side margin of 20%
with a minimum 12 px, and three qualifying frames. Fresh box-local masks add
hull-overlap gating; stale/missing masks skip that condition. `reset()` starts
a new search but preserves the 2 s refractory time. Log every contact effect;
its `event='fire'` lets the trial owner capture `contact_fired_s` relative to
trial start. Operator touch keys remain a separate outcome.

## Speech engine and devices

`SayEngine(cfg).synthesize(text, sample_rate)` (also `.render`) returns float32
mono samples and opens no playback device. `pre_render(texts, sample_rate)`
populates an in-memory cache. Invoke it in the speech worker, not the callback.
The cache includes text, voice, rate and trim settings, and rejects empty/silent
synthesis. The default macOS voice may be slow on a cache miss. Mix samples
through the cue output; cache retrieval is not a measurement of playback start.

Import the standalone engine from `reach.say_engine`. The integrated speech worker exposes its `SayEngine` adapter through `reach.tts`. Pre-render
actual found-position variants including 'back toward you', 'back on your left'
and 'back on your right', as well as the forward/left/right phrases.

On macOS, `speech_devices.mac_builtin_input(devices)` selects the built-in mic.
After opening it, call `stereo_warning(output_info, input_name=mic_name)`.
A known AirPods input/output pairing warns even when Core Audio still reports
2 channels/48 kHz. Device metadata does not prove panning by ear. Keep the
built-in mic and AirPods output in local configuration.

## Offline contact inspection

`reach replay-contact RUN_DIR --frames` derives a CSV and copies each firing
frame locally. `--force` replaces derived outputs only. The input log is never
changed. This box replay cannot reconstruct propagated mask overlap from target
box events, so it is not exact decision replay or physical touch ground truth.
Legacy runs also lack landmarks. Preserve single-rater spot checks and failed
or indeterminate cases.

The integrated live runner, voice client and speech mixer consume these
interfaces. Startup waits for the speech cache before opening the microphone.
Libraries and offline replay are usable independently; live voice/contact
success and endurance are measured separately.

Confirmation, multi-instance semantics and the exact replay record are described in [confirmation.md](confirmation.md).
