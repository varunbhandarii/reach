# How we evaluated Reach

## Who tested it

The only people who used Reach were its two builders, Varun Bhandari and
Chaitanya Manik, who are sighted. For eyes-covered trials we covered our eyes
before reaching and uncovered them only after reporting the touch.

> The only human testers were the two people who built Reach, with their eyes
> covered. That validates the control loop and the audio encoding with a person in
> it; it does not show usability for blind or low-vision users. Designers know the
> cues and the system, which inflates performance. Sighted people also lack trained
> tactile search strategies, and a published comparison found blindfolded sighted
> participants scored higher than blind participants ([Qiu et al.](https://rauterberg.employee.id.tue.nl/publications/BIT2023preprint.pdf)).
> A systematic review of 85 assistive-AI papers found only 16 included participants
> with visual impairment, and calls blindfolded-only evaluation inappropriate as
> the sole evidence ([arXiv 2407.12035](https://arxiv.org/abs/2407.12035)). For that
> reason most of our evidence is participant-free and reproducible from the logs.
> Testing with blind and low-vision users is the required next step.

## What we measured, and how

- **Spoken commands.** A scripted list of 20 requests (find each object, stop,
  cancel, repeat, recheck, yes, no) spoken in the live app; a command counts as
  correct when the logged command matches the script.
- **Contact.** Ten sighted reaches per setup. After each, the tester rated whether
  the automatic chime came at a real touch. The chime fires when the fingertip is
  inside the target, nearly still relative to it, and the hand covers part of the
  object's mask, for three frames in a row.
- **Eyes-covered trials.** A seeded script chose each target and spoke it. The
  tester pressed a key when ready, covered their eyes, reached by sound alone, and
  reported the touch with a key. A trial succeeds with a reported touch within 45 s;
  times include the reporting delay. Every attempt was kept, including failures.
- **Touch check.** After contact, Muse Spark receives a picture of the target taken
  before the hand arrived and a picture at the touch, and answers whether they show
  the same object. We tested it on saved image pairs (correct and deliberately
  mismatched) and live.
- **Latency.** From timestamps in each session's event log. The camera delay was
  measured once per setup with a flashing on-screen square filmed by the phone.
- **Replay.** Every step of the session state machine logs its inputs and outputs.
  `reach replay` rebuilds the decisions from the log and checks they match exactly.

## What we did not measure

- Testing with blind or low-vision people.
- A comparison against searching by touch without Reach.
- End-to-end acoustic latency from hand movement to sound in the ear.
- A public detection benchmark; our object checks use our own photos.
- Body-worn cameras; both setups hold the phone in one hand.
- Clutter beyond a few objects, very dark rooms, or glass and shiny objects.

## Scope

Reach is a hackathon prototype, not a medical device and not a mobility aid. It is
intended only for locating non-hazardous objects on a tabletop within arm's reach.
It is not for navigation, obstacle avoidance, stairs, hot surfaces, sharp tools,
medication identification, or any safety-critical decision, and it refuses requests
for sharp or hot objects. Detections and touch confirmations are AI estimates and
can be wrong; always confirm by touch. Reach does not replace a cane, a guide dog,
or a sighted assistant.

## Privacy

- Camera frames and crops are sent only to the Meta Model API standard tier, whose
  terms state content is not used to train Meta models. Muse Spark requests set
  `store: false`. We never use contributor-tier models with camera frames.
- Microphone audio is streamed to Muse Voice Transcribe for recognition and is not
  saved. Natural-language requests that the local grammar cannot parse are sent to
  Muse Spark as text.
- Session logs, frames and recordings stay on the laptop in an ignored `runs/`
  folder. This repository publishes one events-only session with no frames, audio,
  raw transcripts or device names.
- Only the two of us appear in our recordings.

Two builders testing their own prototype is engineering validation, not research
designed to produce generalizable knowledge, so it did not require ethics board
review. A study with blind or low-vision participants would go through Stony Brook
University's IRB first.
