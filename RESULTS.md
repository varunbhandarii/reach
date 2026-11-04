# Results

Measured between Oct 27 and Nov 4, 2026 on two setups. Every number comes from a
logged run; raw camera frames and audio stay on our laptops, and one events-only
session is published in [`sample_trial/`](sample_trial/). How we tested, and what
these numbers do not show, is in [EVALUATION.md](EVALUATION.md).

| Setup | Laptop | Camera, held in one hand | Audio |
|---|---|---|---|
| **A** | Windows 11 | Android phone, IP Webcam over Wi-Fi, 640×480 | Wired headphones, laptop microphone |
| **B** | MacBook Air | iPhone through Continuity Camera | AirPods Pro (Bluetooth), MacBook microphone |

## Headline results

| Measure | Setup A | Setup B |
|---|---|---|
| Spoken commands understood (scripted list of 20) | **20/20** | **20/20** (reported by the tester) |
| End of speech to command, median (p95) | **531 ms** (553 ms), 5 turns | **515 ms** (657 ms), 53 turns |
| Natural requests understood via Muse Spark, e.g. "can you help me find my reading glasses" | 9/10 | not run |
| Sighted reaches where the contact chime was rated correct | **10/10** | **10/10** |
| Eyes-covered reaches, touch within 45 s | **5/5** (wallet and pen; 11.4 to 18.8 s, median 14.4 s), repeated 5/5 in a second set | not completed with stereo confirmed |
| Two-image touch check on saved pairs: same object confirmed | **30/30** | **30/30** |
| Two-image touch check on saved pairs: different object rejected | **15/15** | **15/15** |
| Touch check, live correct touches | 9 of 10 confirmed (1 time-out) | 10 of 10 confirmed (single-image version) |
| Touch check, live wrong-object touches caught | 4 of 5 | not run |
| 10-minute sessions without an audio dropout | yes (wired, before voice was added) | yes (Bluetooth, voice open, 0 reconnects) |
| Recorded sessions replayed exactly (no camera, model or network) | yes, e.g. 5,397 steps | yes, e.g. 14,352 steps |
| Safety refusal for sharp or hot objects ("find the knife") | refused, no model request sent | not recorded |

Saved-pair checks use 10 correct and 5 mismatched image pairs, each sent 3 times,
so repeats are not independent trials. Setup A's eyes-covered sets used the two
objects available on that setup; setup B had all four tested objects.

## Latency

| Stage | Setup A | Setup B |
|---|---|---|
| Phone camera to laptop (screen-flash test), median / p90 | 131 / 155 ms | 122 / 146 ms (USB) |
| Hand tracking per frame, median / p95 | 11.2 / 16.7 ms (one hand, 480 px input) | 8.7 / 12.2 ms |
| Target propagation (table motion, hand masked), median / p95 | 13.1 / 19.1 ms | 6.2 / 8.9 ms |
| SAM 3.1 response, median / p95 | 763 / 1,387 ms (62 responses) | 771 / 1,447 ms (253 responses) |
| Muse Spark two-image check (saved pairs), median / p95 | 1,433 / 2,042 ms | 1,835 / 3,427 ms |
| Cached speech, request to first audio block, median / p95 | 3.9 / 7.9 ms | 3.1 / 5.5 ms |
| Reported audio output latency | 22 ms (wired) | 165 ms (Bluetooth) |
| Movement to sound, component estimate, median | about 172 ms | about 300 ms |

The movement-to-sound figure adds the camera median, the software time between a
frame and its cue, half an audio block and the reported output latency. It is an
estimate, not an end-to-end acoustic measurement. SAM results arrive asynchronously:
the target box is carried between results by tracking the table, so a late result
does not stall guidance.

Muse Spark latency varied with load: during the evening of Nov 3 we saw single live
touch checks of 5.6 s and scene lists of 7.7 s, and one scene list timed out. Reach
says "You're touching the …" after 2.5 s without waiting for the check.

## Object detection on our own photos

SAM 3.1 with a one-word phrase, judged by eye on photos of our tables:

| Object | Correct | Notes |
|---|---|---|
| wallet | 10/10 | |
| pen | 8/10 | misses when partly covered |
| ball | 10/10 | |
| apple | 10/10 | |
| mug | 10/10 | |
| bottle | 10/10 | |

These are small samples on our own objects, not a public benchmark.

## Problems we found and fixed

- A phone camera stream over Wi-Fi paused for up to a second and once ended a
  session; Reach now reconnects for up to 15 s and shows "Camera paused,
  reconnecting".
- Using AirPods as the microphone switches them to call audio and removes stereo;
  Reach now checks the output route every 2 s on macOS, mutes the guide and says
  so, and refuses an AirPods microphone.
- A one-image touch check sometimes named the right object differently ("stylus"
  for a pen). The check now compares a picture from before the reach with the touch,
  which gave 0 false rejections in 60 saved correct pairs.
- Recognition once merged a restarted request ("find thefind the wallet"); the
  grammar now uses the object after the last request verb.

## Reproduce

```
uv run reach replay sample_trial      # exact replay, no devices or network
uv run reach report runs/<session>    # metrics for a session you record
uv run pytest                         # includes the sample-trial replay
```
