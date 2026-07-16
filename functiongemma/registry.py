"""A tiny, file-backed model registry  —  the boring-but-essential MLops part.

Every serious pipeline needs an answer to "which model is in production, how was
it built, and how do I roll back?". Heavyweight tools (MLflow, Vertex AI Model
Registry, W&B) all reduce to the same core: **immutable versioned artifacts,
each with metadata and metrics, plus mutable stage pointers** (staging /
production) you can flip and revert.

This is that core in ~80 lines of stdlib, storing a JSON manifest on disk. The
promote step (`pipelines/promote.py`) writes here only when a candidate passes
the evaluation gate, so "production" always points at a model that cleared the
quality bar. No network, no service — perfectly adequate for an edge project and
a faithful model of the real thing.
"""

import json
import os
import time
import uuid


class ModelRegistry:
    """Versioned model metadata with promotable stage pointers."""

    STAGES = ("none", "staging", "production", "archived")

    def __init__(self, root="registry"):
        self.root = root
        self.manifest_path = os.path.join(root, "manifest.json")
        os.makedirs(root, exist_ok=True)
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"models": {}, "stages": {s: None for s in self.STAGES[1:]}}

    def _flush(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)  # atomic write.

    def register(self, name, metrics, artifacts=None, tags=None):
        """Record a new immutable model version. Returns its version id."""
        version = uuid.uuid4().hex[:12]
        self._data["models"][version] = {
            "version": version,
            "name": name,
            "metrics": metrics,
            "artifacts": artifacts or {},
            "tags": tags or {},
            "stage": "none",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._flush()
        return version

    def promote(self, version, stage):
        """Move a version to a stage; any prior occupant is archived."""
        if stage not in self.STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if version not in self._data["models"]:
            raise KeyError(f"unknown version {version!r}")
        if stage in ("staging", "production"):
            prev = self._data["stages"][stage]
            if prev and prev in self._data["models"]:
                self._data["models"][prev]["stage"] = "archived"
            self._data["stages"][stage] = version
        self._data["models"][version]["stage"] = stage
        self._flush()
        return version

    def get(self, version):
        return self._data["models"].get(version)

    def current(self, stage="production"):
        """Return the model record currently pointed at by ``stage``."""
        version = self._data["stages"].get(stage)
        return self.get(version) if version else None

    def list(self):
        """All versions, newest first."""
        return sorted(
            self._data["models"].values(),
            key=lambda m: m["created_at"],
            reverse=True,
        )
