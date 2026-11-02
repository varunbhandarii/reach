"""Explicit JPEG preparation for full-frame scene requests."""

import cv2


def encode_frame(image,long_side=512):
    if image is None or not 32<=long_side<=2048:
        raise ValueError('Invalid scene image or resolution')
    ratio=long_side/max(image.shape[:2])
    image=cv2.resize(image,(max(1,round(image.shape[1]*ratio)),max(1,round(image.shape[0]*ratio))),
        interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:raise ValueError('Cannot encode scene frame')
    return encoded.tobytes()
