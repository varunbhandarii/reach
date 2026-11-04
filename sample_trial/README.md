# Sample trial

A real 36-second session from a Windows laptop with an Android phone held in one
hand: "find the wallet", guidance by sound, the hand briefly leaving the view,
contact, and a two-image check by Muse Spark that confirmed the touched object
("brown wallet", same object as before the reach). Spoken guidance included
"Found the wallet, to your right." and "You're touching the wallet."

This is an events-only copy made with `reach replay RUN --strip`: it keeps every
input and output of the session state machine (hand and target observations,
commands, model results, decisions and effects) and nothing else. It contains no
camera frames, no audio, no raw transcripts and no device names.

Replay it without a camera, microphone or network:

```
uv run reach replay sample_trial
```

Expected output: `replay OK: 964 steps, 1968 effects, 7 state changes, read-backs 1`.
The test suite runs the same check on every change.
