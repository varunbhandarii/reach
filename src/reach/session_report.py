"""Summaries of measured loop metrics and explicitly reported trial outcomes."""

import json
from pathlib import Path

import numpy as np

from .run_metrics import summarize as summarize_loop


def summarize(run_dir):
    root = Path(run_dir)
    summary = summarize_loop(root)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((root/"summary.json").read_text(encoding="utf-8")) if (root/"summary.json").exists() else {}
    events = [json.loads(line) for line in (root/"events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    trials = []
    active = None
    path = []
    for event in events:
        if event["type"] == "trial_start":
            active, path = event["data"], []
        elif event["type"] == "reach_sample" and active:
            path.append(event["data"]["relative_xy"])
        elif event["type"] == "trial_end":
            ratio = None
            if len(path) >= 2:
                initial = float(np.linalg.norm(path[0]))
                if initial > .001:
                    ratio = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()/initial)
            trials.append({**event["data"], "relative_path_length_ratio": ratio})
            active, path = None, []
    summary.update(mount=manifest["mount"], source=manifest["source"], objects=manifest["objects"],
        prediction=manifest["prediction"], trial_mode=manifest.get("trial_mode"), completed=saved.get("completed", False),
        duration_s=saved.get("duration_s"), audio_measured=saved.get("audio_measured", False),
        audio_underflows=saved.get("audio", {}).get("underflows"),
        audio_callback_errors=saved.get("audio", {}).get("callback_errors"),
        capture=saved.get("capture"), trials=trials,
        trial_successes=sum(t["success"] for t in trials),
        motion_to_sound_estimate_ms=saved.get("motion_to_sound_estimate_ms"),
        latency_note=manifest.get("latency_note"),
        path_note="Relative source-image path proxy includes camera/target estimation error; not physical hand distance.")
    return summary
