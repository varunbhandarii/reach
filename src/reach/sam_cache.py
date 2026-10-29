"""Local segmentation cache; read mode never sends an image on a miss."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from .sam_client import SamError
from .types import Instance


class CacheMiss(SamError):
    kind = "cache_miss"


class SamCache:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        sam = cfg.get("sam", {})
        mode = sam.get("cache", "auto")
        self.mode = ("write" if cfg.get("camera", {}).get("source") == "file" else "off") if mode == "auto" else mode
        if self.mode not in ("off", "read", "write"):
            raise ValueError("sam.cache must be auto, off, read or write")
        self.root = Path(sam.get("cache_dir", "runs/sam-cache"))
        self.model = sam.get("model", "sam-3.1")
        self.conf_min = float(sam.get("conf_min", .5))

    def _identity(self, jpeg, phrase, frame):
        return {"version": 1, "jpeg_sha256": sha256(jpeg).hexdigest(), "phrase": phrase,
                "model": self.model, "conf_min": self.conf_min, "width": frame.width, "height": frame.height}

    def _path(self, identity):
        key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.root / (key + ".json")

    def read(self, jpeg, phrase, frame):
        if self.mode == "off":
            return None
        identity = self._identity(jpeg, phrase, frame)
        try:
            data = json.loads(self._path(identity).read_text(encoding="utf-8"))
            if data["identity"] != identity:
                return None
            instances = tuple(Instance(item["sam_id"], tuple(item["box_xyxy"]), item["conf"], item["mask_rle"]) for item in data["instances"])
            for item in instances:
                x0, y0, x1, y1 = item.box_xyxy
                if not (0 <= x0 < x1 <= frame.width and 0 <= y0 < y1 <= frame.height):
                    return None
            return instances
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def write(self, jpeg, phrase, frame, instances):
        if self.mode != "write":
            return
        identity = self._identity(jpeg, phrase, frame)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(identity)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.root, suffix=".tmp", delete=False, encoding="utf-8") as stream:
                temporary = Path(stream.name)
                json.dump({"identity": identity, "instances": [asdict(i) for i in instances]}, stream, allow_nan=False)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
