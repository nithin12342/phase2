"""Assemble the full 200-feature vector plus the binary label per video."""
import csv
from pathlib import Path

from .schema_catalog import CATALOG

FEATURE_IDS = [e[0] for e in CATALOG]


def assemble_rows(records, features_by_pid):
    """Fill the 200-catalog for each record. Missing features are NaN + Absent.

    records: D-Vlog records (video_id, label, ...).
    features_by_pid: {video_id: {feature_id: (value, reliability)}}.
    Returns (wide_rows, reliability_rows).
    """
    wide = []
    rel = []
    for rec in records:
        pid = str(rec["video_id"])
        feats = features_by_pid.get(pid, {})
        row = {"video_id": pid, "label": rec["label"]}
        for fid in FEATURE_IDS:
            if fid in feats:
                val, r = feats[fid]
            else:
                val, r = float("nan"), "Absent"
            row[fid] = val
            rel.append((pid, fid, r))
        wide.append(row)
    return wide, rel


def write_outputs(outdir, wide, rel):
    """Write features_wide.csv, reliability.csv, and labels.csv atomically-ish."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "features_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label", *FEATURE_IDS])
        w.writeheader()
        for row in wide:
            w.writerow(row)
    with open(outdir / "reliability.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "feature_id", "reliability"])
        w.writerows(rel)
    with open(outdir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label"])
        for row in wide:
            w.writerow([row["video_id"], row["label"]])