"""Deterministic SAM client placeholder for component integration."""

from .types import Frame, Instance


async def segment(jpeg: bytes, phrase: str, frame: Frame) -> tuple[Instance, ...]:
    """Return a stable sample detection without making a network request."""
    del jpeg, phrase

    left = frame.width // 4
    top = frame.height // 4
    right = max(left + 1, (frame.width * 3) // 4)
    bottom = max(top + 1, (frame.height * 3) // 4)
    right = min(right, frame.width)
    bottom = min(bottom, frame.height)

    if right <= left or bottom <= top:
        return ()

    return (
        Instance(
            sam_id="1",
            box_xyxy=(left, top, right, bottom),
            conf=0.9,
            mask_rle=None,
        ),
    )
