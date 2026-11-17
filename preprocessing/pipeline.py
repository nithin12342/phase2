"""Batch orchestrator: signals -> Rust features -> 200-wide + embeddings.

Every batch is written to its own folder and recorded in a Drive-side manifest
before the next batch starts. A restart skips finished videos. Nothing critical
lives only on the VM disk.

In run mode a producer thread downloads videos while the main thread processes
them, so network latency hides behind GPU/Rust work. A drive_lock serializes the
Drive flush against downloads: the cloud sync never unmounts mid-write.

Modes:
  sample: synthetic signals + synthetic embeddings. No network, no GPU.
  run:    download -> decode -> ASR -> real signals + GPU encoders (Colab).
"""
from pathlib import Path
import queue
import shutil
import threading

from . import assemble, config, encoders, rust_bridge, sampling, signals
from .manifest import Manifest


def _seed(record):
    vid = str(record.get("video_id") or "0")
    return int(vid) if vid.isdigit() else abs(hash(vid)) % (2 ** 31)


def _signals_for(record, mode, raw_root, workdir, video=None, proxy=None):
    if mode == "sample":
        return signals.synthetic_signals(_seed(record), record["label"] == 1), {}
    from . import asr, media  # lazy: Colab only

    if video is None:
        video = media.download_video(record["key"], raw_root, proxy=proxy)
    frames_dir = config.FRAMES_LOCAL / f"{record['key']}_frames"
    wav, frames, raw_gray = media.extract_media(video, frames_dir)
    transcript = asr.transcribe(wav)
    ctx = {"wav_path": wav, "frame_paths": frames, "text": transcript["text"]}
    sig = signals.derive_signals(record, wav, raw_gray, transcript["text"], workdir)
    return sig, ctx


def _download_producer(todo, raw_root, q, drive_lock, proxy=None):
    """Fetch videos ahead of the consumer. One bad fetch never stops the rest."""
    from . import media  # lazy: Colab only
    try:
        for rec in todo:
            try:
                with drive_lock:
                    video = media.download_video(rec["key"], raw_root, proxy=proxy)
                q.put((rec, video, None))
            except Exception as exc:
                q.put((rec, None, f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(None)  # sentinel: the consumer stops here even on a producer crash


def _iter_queue(q):
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def _process_batch(item_iter, mode, out_root, raw_root, batch_idx, batch_size,
                   proxy=None):
    """Pull up to batch_size items, processing each as it arrives.

    Processing per item (not collect-then-process) lets the producer download
    the next video during this one's GPU/Rust work. Returns
    (workdir, ok_records, failed, exhausted); workdir is None when no item came.
    """
    workdir = Path(out_root) / "batches" / f"batch_{batch_idx:04d}"
    emb_dir = workdir / "embeddings"
    participant_signals = {}
    ok_records = []
    failed = []
    n = 0
    exhausted = False
    for item in item_iter:
        if n == 0:
            workdir.mkdir(parents=True, exist_ok=True)
        rec, video, err = item
        pid = str(rec["video_id"])
        if err is not None:
            failed.append((pid, err))
        else:
            try:
                sig, ctx = _signals_for(rec, mode, raw_root, workdir, video=video,
                                        proxy=proxy)
                seed = _seed(rec) if mode == "sample" else None
                emb = encoders.encode_all(ctx, synthetic_seed=seed)
                encoders.save_embeddings(emb_dir, pid, emb)
                frames = ctx.get("frame_paths") or []
                if frames:
                    shutil.rmtree(Path(frames[0]).parent, ignore_errors=True)
                participant_signals[pid] = sig
                ok_records.append(rec)
            except Exception as exc:  # one bad video must not abort the batch
                failed.append((pid, f"{type(exc).__name__}: {exc}"))
        n += 1
        if n >= batch_size:
            break
    else:
        exhausted = True
    if n == 0:
        return None, [], [], True

    if participant_signals:
        features, _ = rust_bridge.compute_features(participant_signals, workdir)
    else:
        features = {}
    wide, rel = assemble.assemble_rows(ok_records, features)
    assemble.write_outputs(workdir, wide, rel)
    return workdir, ok_records, failed, exhausted


def _next_batch_index(out_root):
    """First unused batch index, so a resume never overwrites a saved batch."""
    idxs = []
    for b in (Path(out_root) / "batches").glob("batch_*"):
        suffix = b.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            idxs.append(int(suffix))
    return max(idxs) + 1 if idxs else 0


def merge_batches(out_root, records):
    """Concatenate every batch folder into the final wide/reliability/labels."""
    import csv

    out_root = Path(out_root)
    batches = sorted((out_root / "batches").glob("batch_*"))
    wide, rel = [], []
    for b in batches:
        if not (b / "features_wide.csv").exists():
            continue  # an interrupted batch leaves a folder with no CSV yet
        with open(b / "features_wide.csv", newline="", encoding="utf-8") as f:
            wide.extend(csv.DictReader(f))
        with open(b / "reliability.csv", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the per-batch header
            rel.extend(tuple(r) for r in reader)
    assemble.write_outputs(out_root, wide, rel)
    return len(wide)


def run(records, out_root, mode="sample", batch_size=8, limit=None, raw_root=None,
        flush=None, proxy=None):
    """Process records in resumable batches. Returns a summary dict.

    flush(out_root), when given, runs after each batch and after the final
    merge. On Colab it blocks until Drive uploads reach the cloud, so a runtime
    reset never loses a finished batch. The local sample passes no flush.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = Path(raw_root) if raw_root else out_root / "raw"
    manifest = Manifest(out_root / "manifest.json")

    # A pilot subset is class-balanced; limit=None keeps the full dataset.
    records = sampling.stratified_subset(records, limit)
    todo = [r for r in records
            if not manifest.is_done(str(r["video_id"]))
            and not manifest.is_failed(str(r["video_id"]))]

    # The flush unmounts Drive; the producer writes to it. One lock keeps them
    # from overlapping. Processing holds no lock, so it overlaps downloads.
    drive_lock = threading.RLock()
    locked_flush = None
    if flush:
        def locked_flush(root):
            with drive_lock:
                flush(root)

    producer = None
    if mode == "run" and todo:
        q = queue.Queue(maxsize=batch_size)
        producer = threading.Thread(
            target=_download_producer, args=(todo, raw_root, q, drive_lock, proxy),
            daemon=True)
        producer.start()
        item_iter = _iter_queue(q)
    else:
        item_iter = ((rec, None, None) for rec in todo)

    processed = 0
    failed_count = 0
    base_idx = _next_batch_index(out_root)
    batch_no = 0
    while True:
        workdir, ok_records, failed, exhausted = _process_batch(
            item_iter, mode, out_root, raw_root, base_idx + batch_no, batch_size,
            proxy=proxy)
        if workdir is not None:
            for rec in ok_records:
                manifest.mark(str(rec["video_id"]), {
                    "label": rec["label"], "batch": workdir.name, "mode": mode})
            for pid, err in failed:
                manifest.mark_failed(pid, {
                    "error": err, "batch": workdir.name, "mode": mode})
            processed += len(ok_records)
            failed_count += len(failed)
            if locked_flush:
                locked_flush(out_root)
            batch_no += 1
        if exhausted:
            break
    if producer is not None:
        producer.join()

    total = merge_batches(out_root, records)
    if locked_flush:
        locked_flush(out_root)
    return {
        "mode": mode,
        "requested": len(records),
        "skipped_done": len(records) - len(todo),
        "processed": processed,
        "failed": failed_count,
        "merged_rows": total,
        "manifest_done": len(manifest),
        "manifest_failed": len(manifest.failed_keys()),
        "out_root": str(out_root),
    }