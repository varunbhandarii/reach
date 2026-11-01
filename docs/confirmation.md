# Confirmation, selection and deterministic replay

The reducer accepts `results=tuple[ReadBack, ...]` alongside observations and commands. Every timer uses the caller's `now`; it performs no device or network work.

Contact plays its chime immediately and enters CONFIRMING. A `readback` effect carries `request_id`, `frame_id`, `box_xyxy` and `phrase`. The frame id names the target observation's source geometry: fetch that frame from the camera ring and construct the crop before handing it to the worker. Keep returned request id, target frame id and phrase unchanged. Stopped, replaced, unmatched and expired results are ignored.

The runner adapter drains the worker's `poll()` before `step` and dispatches `readback` to its nonblocking `request`. Configure `spark.enabled=false` for contact-only operation; it emits the chime and touching fallback without an API request. The library implementation and current instance-effect dispatcher do not themselves supply a cloud read-back worker. Integrate that worker before claiming live verification or using the read-back CLI options.

- At a confident match within 2.5 seconds, say “You have the X.” The optional hint follows at lower priority.
- At 2.5 seconds without an answer, say the touching fallback once. A later match adds no confirmation; a confident mismatch can still recover until the 6-second deadline.
- A mismatch excludes the old box, then selects another candidate or searches again. A second mismatch in the same search ends it. Error/unsure results and deadline expiry finish with the touching fallback.
- The confidence threshold defaults to 0.6 for a definite match or mismatch. Uncertain output cannot produce “You have the X.”
- `brain.ask_touch_enabled` optionally asks about a fingertip held inside for 1.5 seconds. It defaults off. A yes within five seconds enters confirmation; no closes that window. This is separate from ordinary cached yes/no acknowledgements.

Perception publishes at most three candidates. Only the chosen candidate has a mask; other boxes propagate cheaply. `choose(sam_id)` changes the lock. `exclude(frame_id, box_xyxy)` maps the rejected box forward and filters new detections at IoU 0.3 or above. A new phrase or cancellation clears exclusions; a same-phrase retry preserves them. IDLE pauses query/publishing while preserving motion hypotheses for a read-back reply; `set_phrase(None)` ends the search. A late exclusion can use the paused selected lock even after short transform history expires. If mapping is unavailable it is logged as unavailable, never guessed.

The controller asks for a spatial choice when two or more candidates are within `sam.tie_margin`. Left/right compare horizontal positions; closer/farther compare vertical positions, with flip respected. Spoken “ahead” parses to farther. Invalid choices prompt once. After eight seconds, choose the nearest candidate to the fingertip, or perception's choice if no hand is visible.

## Replay record

Store `manifest.config = session_codec.replay_config(cfg, objects)` and the object list. This retains reducer settings without credentials, endpoints or audio-device names. The current manifest seam does this automatically. Do not substitute a config hash for the actual settings.

The first effect of every step is a log with `key='brain_in'`. Its payload is a plain-JSON snapshot of seq, now, hand, target including masks, the perception-stat fields consumed, commands, read-back results, speaking, frame size and flip. The last effect logs `brain_out` with the returned decision and ordered domain effects. Its effects omit the two logging wrappers to avoid recursive records. Replay reconstructs both wrappers and compares the complete returned effect list. The runner must log these payloads without unwrapping or editing fields, and log the independent decision event as usual.

```sh
reach replay runs/example
reach replay runs/example --strip runs/example-events-only
```

The command prints `replay OK` with step/effect/state/read-back counts, or the first differing step, field, logged value and replayed value. A legacy run prints `no replay inputs`; that is not an exact-replay pass. Missing output records and malformed logs are errors.

Stripping first verifies the original trace, leaves it unchanged, and refuses to overwrite an existing output directory. It retains only manifest settings and replay input/output/decision events. Command transcripts become canonical parsed commands and device identities are removed. Noun labels, read-back descriptions and mask geometry remain because replay consumes them; inspect those before publishing. There are no image or audio files. The stripped copy replays independently.

Exact replay verifies controller behavior under recorded inputs. It does not prove camera accuracy, physical touch, correct recognition or audible output. Contact replay remains available for legacy box/firing inspection.
