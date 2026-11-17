"""Per-video signals for the Rust feature CLI.

Two sources:
  * synthetic_signals: deterministic fake traces for the local sample run.
  * derive_signals: real f0 + motion from the std-only Rust kernels, plus
    lexical counts and timestamps from the ASR text.

The local sample mode never touches media or the CLI. Signal names match the
Rust read_signals_csv contract: f0, motion, au12, unique, total, first_ts,
last_ts, speech_ratio.
"""
import random

from .asr import token_counts

SIGNAL_NAMES = ("f0", "motion", "au12", "unique", "total", "first_ts", "last_ts",
                "speech_ratio")

# Approximate F0 frames per second, used only for the speech_ratio proxy.
F0_FRAME_RATE = 50.0


def synthetic_signals(seed, depressed):
    """Deterministic signals that differ by class. No media, no heavy deps.

    Depressed voices: flatter pitch, more stillness, less smile (AU12), and
    lower lexical diversity. Normal voices: the opposite.
    """
    rng = random.Random(seed)
    n = 64
    if depressed:
        base, spread = 120.0, 6.0
        motion_hi, au_hi = 0.4, 0.05
        unique, total = 40, 200
        speech_ratio = 0.3
    else:
        base, spread = 180.0, 30.0
        motion_hi, au_hi = 1.6, 0.6
        unique, total = 120, 200
        speech_ratio = 0.7
    f0 = [base + rng.uniform(-spread, spread) for _ in range(n)]
    motion = [rng.uniform(0.0, motion_hi) for _ in range(n)]
    au12 = [rng.uniform(0.0, au_hi) for _ in range(n)]
    return {
        "f0": f0,
        "motion": motion,
        "au12": au12,
        "unique": float(unique),
        "total": float(total),
        "first_ts": 0.0,
        "last_ts": float(n) / 8.0,
        "speech_ratio": speech_ratio,
    }


def derive_signals(record, wav_path=None, raw_gray=None, text="", workdir=None):
    """Build the signal dict from real decoded media via the Rust kernels.

    f0 + motion come from the std-only Rust CLI (rust_bridge.derive_signals),
    which replaces librosa.pyin and cv2. Any missing input yields an empty
    trace; the Rust feature side then emits NaN + Absent. AU12 needs OpenFace;
    it stays absent until that adapter is wired on Colab.
    """
    from . import rust_bridge  # lazy: Colab run path only

    signals = {name: [] for name in ("f0", "motion", "au12")}
    pid = str(record.get("video_id") or "0")
    if wav_path is not None and raw_gray is not None and workdir is not None:
        derived = rust_bridge.derive_signals(wav_path, raw_gray, workdir, pid)
        signals["f0"] = derived["f0"]
        signals["motion"] = derived["motion"]
    counts = token_counts(text or "")
    signals["unique"] = float(counts["unique"])
    signals["total"] = float(counts["total"])
    dur = float(record.get("duration") or 0.0)
    signals["first_ts"] = 0.0
    signals["last_ts"] = dur
    if signals["f0"] and dur > 0.0:
        signals["speech_ratio"] = min(1.0, len(signals["f0"]) / (dur * F0_FRAME_RATE))
    return signals