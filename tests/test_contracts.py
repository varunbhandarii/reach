"""JSON compatibility checks for shared data contracts."""

import json
from dataclasses import asdict, fields, is_dataclass

import pytest

from reach.types import (
    Command,
    CueParams,
    Decision,
    Frame,
    Effect,
    HandObs,
    Instance,
    TargetObs,
    ReadBack,
    SceneList,
)


def _restore(value, original):
    if is_dataclass(original):
        return type(original)(
            **{
                field.name: _restore(value[field.name], getattr(original, field.name))
                for field in fields(original)
            }
        )
    if isinstance(original, tuple):
        return tuple(_restore(item, source) for item, source in zip(value, original, strict=True))
    if isinstance(original, dict):
        return {key: _restore(value[key], item) for key, item in original.items()}
    return value


@pytest.mark.parametrize(
    "sample",
    [
        ReadBack(7, 3, "wallet", True, "wallet", "", .9, 123.0),
        ReadBack(8, 4, "pen", None, None, None, None, 6000.0, "timeout"),
        Effect("readback", payload={"request_id": 7, "frame_id": 3, "box_xyxy": [1,2,3,4], "phrase": "wallet"}),
        Effect("choose_instance", payload={"sam_id": "second"}),
        Effect("exclude_instance", payload={"frame_id": 3, "box_xyxy": [1,2,3,4]}),
        Effect("speak", "Looking for the wallet.", "looking", 1),
        Frame(3, 1.25, 1_777_000_000.5, 640, 480, "abc123", "iphone_continuity"),
        HandObs(3, True, (12.5, 22.5), (15.0, 24.0), ((1.0, 2.0), (3.0, 4.0)), (4.0, -2.0), 0.98),
        Instance("14", (10, 20, 30, 40), 0.86, {"size": [20, 20], "counts": "abc"}),
        TargetObs(3, "blue mug", (Instance("14", (10, 20, 30, 40), 0.86, None),), 0, "sam", 12.0),
        Command("find", "blue mug", "find the blue mug", 7, 2.5),
        CueParams(0.5, 440.0, 4.0, False, 0.2, 0.3),
        Decision(11, 3.5, "GUIDING", 0.1, -0.2, 0.3, CueParams(0.5, 440.0, 4.0, False, 0.2, 0.3), None, "found"),
    ],
    ids=["readback", "readback-error", "readback-effect", "choose-effect", "exclude-effect", "effect", "frame", "hand-observation", "instance", "target-observation", "command", "cue", "decision"],
)
def test_contract_round_trips_through_json(sample):
    encoded = json.dumps(asdict(sample), ensure_ascii=False)
    decoded = json.loads(encoded)

    assert _restore(decoded, sample) == sample


def test_scene_and_optional_command_fields_restore_through_session_codec():
    from reach.session_codec import plain,restore_inputs
    scene=SceneList(7,10,(('pen','left'),),12)
    command=Command('find','pen','pen',1,0,'spark_parse','hi')
    result=restore_inputs({'results':[plain(scene)],'commands':[plain(command)]})
    assert result['results']==(scene,) and result['commands']==(command,)
