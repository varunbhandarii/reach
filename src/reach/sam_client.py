"""Synchronous SAM streaming client; no retries, credentials or payload logging."""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

from .masks import encode_rle
from .types import Frame, Instance


class SamError(RuntimeError):
    kind = "sam"


class SamRateLimited(SamError):
    kind = "rate_limited"

    def __init__(self, retry_after=1.0):
        self.retry_after = retry_after
        super().__init__("SAM rate limited; retry after the backoff interval")


class SamPermanentError(SamError):
    kind = "permanent"


class SamNetworkError(SamError):
    kind = "network"


class SamTimeout(SamError):
    kind = "timeout"


class SamParseError(SamError):
    kind = "parse"


def retry_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 1.0
    return max(0.0, seconds) if math.isfinite(seconds) else 1.0


def parse_instances(text: str, frame: Frame, conf_min=.5) -> tuple[Instance, ...]:
    try:
        parser = image_segmentation_format().create_parser()
        parser.push(text, emit=False)
        result = parser.finish(CompletedOutcome()).result
        if any(d.severity == "error" for d in result.diagnostics):
            raise ValueError("parser diagnostics")
        boxes = [r for r in result.records if r.kind == "box"]
        masks = [r for r in result.records if r.kind == "mask"]
        by_id = {r.object_id: r for r in masks}
        sizes = re.findall(r"<\|box;[^|]*;w=(\d+);h=(\d+)", text)
        if (any(r.kind == "text" for r in result.records) or len(boxes) != len(sizes)
                or len(boxes) != len(masks) or len(by_id) != len(masks)
                or len({r.object_id for r in boxes}) != len(boxes)
                or any((int(w), int(h)) != (frame.width, frame.height) for w, h in sizes)):
            raise ValueError("invalid image records")
        instances = []
        for box in boxes:
            mask = by_id[box.object_id]
            if box.confidence is None or mask.confidence is None or mask.mask.encoding != "one_bit":
                raise ValueError("missing confidence or incorrect mask encoding")
            confidence = float(box.confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid confidence")
            if confidence < conf_min:
                continue
            x0, y0 = math.floor(box.left), math.floor(box.top)
            x1, y1 = math.ceil(box.right), math.ceil(box.bottom)
            if not (0 <= x0 < x1 <= frame.width and 0 <= y0 < y1 <= frame.height):
                raise ValueError("box outside source")
            raster = np.frombuffer(decode_mask_to_raster(mask.mask), np.uint8).reshape(mask.mask.height, mask.mask.width)
            raster = cv2.resize(raster, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST_EXACT)
            instances.append(Instance(str(box.object_id), (x0, y0, x1, y1), confidence, encode_rle(raster)))
        return tuple(instances)
    except Exception:
        raise SamParseError("SAM returned invalid segmentation data") from None


class SamClient:
    def __init__(self, cfg=None, *, api_key=None):
        sam = (cfg or {}).get("sam", {})
        self.model = sam.get("model", "sam-3.1")
        if self.model != "sam-3.1":
            raise ValueError("only standard sam-3.1 is supported")
        self.conf_min = float(sam.get("conf_min", .5))
        self.deadline_s = float(sam.get("deadline_s", 3))
        if not 0 <= self.conf_min <= 1 or not 0 < self.deadline_s <= 3:
            raise ValueError("invalid confidence or deadline")
        self._key = api_key

    def segment(self, jpeg: bytes, phrase: str, frame: Frame) -> tuple[Instance, ...]:
        if not jpeg or not phrase.strip():
            raise SamPermanentError("SAM requires an image and a noun phrase")
        load_dotenv()
        key = self._key or os.environ.get("MODEL_API_KEY")
        if not key:
            raise SamPermanentError("MODEL_API_KEY is not configured")
        body = json.dumps({"model": self.model, "stream": True,
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": phrase},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}]}]}).encode()
        start = time.perf_counter()
        connection = http.client.HTTPSConnection("api.meta.ai", timeout=self.deadline_s)
        expired = threading.Event()

        def abort():
            expired.set()
            sock = connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

        timer = threading.Timer(self.deadline_s, abort)
        timer.daemon = True
        timer.start()
        try:
            connection.request("POST", "/v1/responses", body, {"Authorization": "Bearer " + key,
                "Content-Type": "application/json", "Accept": "text/event-stream"})
            response = connection.getresponse()
            if response.status == 429:
                raise SamRateLimited(retry_seconds(response.getheader("Retry-After")))
            if 400 <= response.status < 500:
                raise SamPermanentError(f"SAM rejected the request (HTTP {response.status})")
            if response.status != 200:
                raise SamNetworkError("SAM server is unavailable")
            parts, text, complete = [], [], False
            while True:
                if expired.is_set():
                    raise SamTimeout("SAM exceeded its total deadline")
                raw = response.readline(2**20)
                if not raw:
                    break
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    parts.append(line[5:].lstrip())
                elif not line and parts:
                    data, parts = "\n".join(parts), []
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    kind = event.get("type")
                    if kind == "response.output_text.delta":
                        text.append(event["delta"])
                    elif kind == "response.completed":
                        complete = True
                        break
                    elif kind in ("error", "response.failed", "response.incomplete"):
                        raise SamNetworkError("SAM stream did not complete")
            if not complete:
                raise SamNetworkError("SAM stream closed without completion")
            result = parse_instances("".join(text), frame, self.conf_min)
            if expired.is_set() or time.perf_counter()-start > self.deadline_s:
                raise SamTimeout("SAM exceeded its total deadline")
            return result
        except SamError:
            if expired.is_set():
                raise SamTimeout("SAM exceeded its total deadline") from None
            raise
        except (TimeoutError, socket.timeout):
            raise SamTimeout("SAM exceeded its total deadline") from None
        except (ValueError, UnicodeError, KeyError, TypeError):
            raise SamParseError("SAM returned an invalid event stream") from None
        except (OSError, http.client.HTTPException):
            if expired.is_set():
                raise SamTimeout("SAM exceeded its total deadline") from None
            raise SamNetworkError("SAM connection failed") from None
        finally:
            timer.cancel()
            connection.close()


def segment(jpeg: bytes, phrase: str, frame: Frame) -> tuple[Instance, ...]:
    return SamClient().segment(jpeg, phrase, frame)
