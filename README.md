# Reach

Voice-guided hand-to-object finding for blind and low-vision people, using a phone camera, Meta's SAM 3.1, and stereo sound cues.

> **Status:** in development. Built during the Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026.

Current camera experiments include a handheld phone, with one hand holding the camera and the other reaching. A chest mount is available for the iPhone setup; stable body-worn operation has not yet been validated.

Reach is a prototype for locating non-hazardous objects on a tabletop within arm's reach. It is not a medical device and not a mobility aid.

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. Works on Windows and macOS.

```
uv python install 3.12
uv sync
uv run pytest
```

Copy `.env.example` to `.env` and add your own Meta Model API key. Never commit `.env`.

## License

Apache-2.0. See [LICENSE](LICENSE).
