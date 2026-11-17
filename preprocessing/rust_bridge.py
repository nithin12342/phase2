"""Bridge to the Rust feature CLI via stdlib subprocess.

Python writes long-format signals; Rust computes the scalar features and
returns value + reliability. This keeps the math in Rust (the 1:100 rule).
"""
import csv
import subprocess
from pathlib import Path

from . import config


def find_cli():
    """Locate the built Rust CLI (release first, then debug; .exe on Windows)."""
    candidates = [
        config.RUST_CLI, config.RUST_CLI_DEBUG,
        config.RUST_CLI.with_suffix(".exe"),
        config.RUST_CLI_DEBUG.with_suffix(".exe"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError("Rust CLI not built. Run `cargo build` in rust/.")


def _write_signals(path, participant_signals):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["participant", "signal", "values"])
        for pid, signals in participant_signals.items():
            for name, val in signals.items():
                vals = list(val) if isinstance(val, (list, tuple)) else [val]
                w.writerow([pid, name, *vals])


def compute_features(participant_signals, workdir):
    """Compute features for many participants in one Rust call.

    participant_signals: {pid: {signal_name: list_or_scalar}}.
    Returns {pid: {feature_id: (value, reliability)}}.
    """
    cli = find_cli()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    sig_path = workdir / "signals.csv"
    out_path = workdir / "features.csv"
    _write_signals(sig_path, participant_signals)
    proc = subprocess.run(
        [str(cli), "--features", str(sig_path), str(out_path)],
        check=True, capture_output=True, text=True,
    )
    result = {}
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["value"]
            val = float("nan") if raw in ("NaN", "", "nan") else float(raw)
            result.setdefault(row["participant"], {})[row["id"]] = (
                val, row["reliability"])
    return result, proc.stdout.strip()


def derive_signals(wav_path, raw_gray, workdir, pid, size=64):
    """Derive f0 + motion traces from raw media via the Rust CLI (std-only).

    Replaces librosa.pyin (f0) and cv2 frame-diff (motion). The wav is 16-bit
    PCM; raw_gray is a grayscale size*size plane stream from ffmpeg rawvideo.
    Returns {"f0": [...], "motion": [...]}.
    """
    cli = find_cli()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"derived_{pid}.csv"
    subprocess.run(
        [str(cli), "--derive", str(wav_path), str(raw_gray),
         str(size), str(size), str(pid), str(out_path)],
        check=True, capture_output=True, text=True,
    )
    result = {"f0": [], "motion": []}
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] == "participant":
                continue
            name = row[1]
            if name in result:
                result[name] = [float(v) for v in row[2:]
                                if v not in ("", "NaN", "nan")]
    return result