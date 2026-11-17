"""Read the D-Vlog metadata CSV into binary-label records."""
import csv

from .config import LABEL_MAP


def load_records(csv_path):
    """Return one record per video with a binary depression label.

    Columns: video_id, label, key, gender, duration, channelId.
    The YouTube video ID is the `key` column.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = LABEL_MAP.get((row.get("label") or "").strip().lower())
            if label is None:
                continue
            records.append({
                "video_id": (row.get("video_id") or "").strip(),
                "key": (row.get("key") or "").strip(),
                "label": label,
                "gender": (row.get("gender") or "").strip(),
                "duration": float(row.get("duration") or 0.0),
                "channel": (row.get("channelId") or "").strip(),
            })
    return records


def label_counts(records):
    """Return {0: n_normal, 1: n_depression}."""
    counts = {0: 0, 1: 0}
    for r in records:
        counts[r["label"]] += 1
    return counts