"""Box-local binary masks in uncompressed, column-major COCO RLE."""

import cv2
import numpy as np


def encode_rle(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mask must be two dimensional")
    flat = (mask != 0).ravel(order="F")
    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], cuts, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def decode_rle(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    if h < 0 or w < 0 or any(not isinstance(c, int) or c < 0 for c in counts) or sum(counts) != h*w:
        raise ValueError("invalid mask runs")
    return np.repeat((np.arange(len(counts)) % 2).astype(np.uint8), counts).reshape((h, w), order="F")


def resize_rle(rle, width, height):
    return encode_rle(cv2.resize(decode_rle(rle), (width, height), interpolation=cv2.INTER_NEAREST_EXACT))
