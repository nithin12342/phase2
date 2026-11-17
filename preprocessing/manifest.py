"""Drive-side manifest with atomic writes for idempotent batch resume."""
import json
import os
from pathlib import Path


class Manifest:
    """Tracks completed keys. A restart skips already-saved work."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            data = {"version": 1, "done": {}}
        data.setdefault("failed", {})
        return data

    def is_done(self, key):
        return str(key) in self._data["done"]

    def is_failed(self, key):
        return str(key) in self._data["failed"]

    def mark(self, key, info):
        self._data["done"][str(key)] = info
        self._data["failed"].pop(str(key), None)
        self._save()

    def mark_failed(self, key, info):
        self._data["failed"][str(key)] = info
        self._save()

    def failed_keys(self):
        return list(self._data["failed"])

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def done_keys(self):
        return list(self._data["done"])

    def __len__(self):
        return len(self._data["done"])