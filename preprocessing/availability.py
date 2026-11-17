"""Classify each D-Vlog video as public or private with a metadata-only request.

The program reads video links from one file (the dataset CSV) and account
cookies from another file. It asks YouTube for each video's metadata, never
the media bytes. One yt-dlp session is reused for every video.

The result is a resumable CSV on Drive. Each checkpoint writes atomically
(temp file -> rename) and flushes Drive, so a VM reset never loses work.

Cookie values are credentials. This module never prints them. It passes the
cookie file path to yt-dlp and reports only the path on a missing-file error.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from . import config, dvlog
from .media import video_url

HEADER = ["video_id", "key", "label", "status", "detail"]

# Verdicts that allow a download with account cookies. Every other verdict is
# unreachable (private, removed, blocked) or needs extra auth.
PROCESSABLE = ("public", "unlisted")


def _clean(text, width=200):
    """One single-line, length-capped string for the detail column."""
    return " ".join(str(text).split())[:width]


def _build_opts(cookies, node_bin, sample_url):
    """yt-dlp options from the CLI parser, identical to the download path.

    The parser fills the defaults that pass the YouTube player bot check.
    A hand-built dict misses them and fails with LOGIN_REQUIRED. The mweb
    client plus account cookies plus the node EJS solver are all required.
    """
    cookies = Path(cookies)
    if not cookies.exists():
        raise FileNotFoundError(f"cookie file not found: {cookies}")
    import yt_dlp  # lazy: Colab only
    argv = [
        "--cookies", str(cookies),
        "--js-runtimes", f"node:{node_bin}",
        "--remote-components", "ejs:github",
        "--extractor-args", "youtube:player_client=mweb",
        "--format", "18/b",
        "--no-warnings",
        sample_url,
    ]
    parsed = yt_dlp.parse_options(argv)
    opts = next(x for x in parsed if isinstance(x, dict) and "format" in x)
    opts["skip_download"] = True
    # Raise DownloadError instead of returning None, so _classify sees the reason.
    opts["ignoreerrors"] = False
    opts["quiet"] = True
    opts["noprogress"] = True
    return opts


def _from_error(err):
    """Map a yt-dlp DownloadError message to a status label."""
    msg = str(err).lower()
    detail = _clean(err)
    if "private" in msg:
        return "private", detail
    if "members" in msg or "join this channel" in msg:
        return "members", detail
    if "confirm your age" in msg or "inappropriate" in msg or "age-restricted" in msg:
        return "age-restricted", detail
    if "your country" in msg or "not made this video available" in msg:
        return "geo-blocked", detail
    if "copyright" in msg or "blocked it" in msg:
        return "blocked", detail
    if any(w in msg for w in ("removed", "deleted", "terminated",
                              "unavailable", "no longer available")):
        return "unavailable", detail
    if "sign in" in msg or "login" in msg or "cookies" in msg:
        return "login-required", detail
    return "error", detail


def _from_info(info):
    """Map a successful metadata response to a status label and title."""
    avail = (info.get("availability") or "public").strip().lower()
    mapping = {
        "public": "public", "unlisted": "unlisted", "private": "private",
        "needs_auth": "login-required", "subscriber_only": "members",
        "premium_only": "premium",
    }
    status = mapping.get(avail, avail)
    return status, _clean(info.get("title") or "", 80)


def _classify(ydl, key):
    """Request one video's metadata. Return (status, detail). No bytes fetched."""
    import yt_dlp
    try:
        info = ydl.extract_info(video_url(key), download=False)
    except yt_dlp.utils.DownloadError as err:
        return _from_error(err)
    except Exception as err:  # network or extractor fault: keep the census going
        return "error", _clean(err)
    if not info:
        return "error", "no metadata returned"
    return _from_info(info)


def _load_rows(out_csv):
    """Existing result rows, so a resumed run keeps prior classifications."""
    out_csv = Path(out_csv)
    if not out_csv.exists():
        return []
    with open(out_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("key")]


def _atomic_write(out_csv, rows, fields=HEADER):
    """Write all rows to a temp file, then rename over the target."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(out_csv)


def _drive_flush(out_csv):
    """Block until the CSV reaches the cloud, then remount and verify."""
    from google.colab import drive
    drive.flush_and_unmount()
    drive.mount("/content/drive")
    if not Path(out_csv).exists():
        raise RuntimeError(f"Drive flush lost the census file: {out_csv}")


def census(records, cookies, out_csv, node_bin=None, delay=1.0,
           limit=None, flush=None, checkpoint=25):
    """Classify every record not already settled. Resume-safe and Drive-durable.

    A prior verdict is kept unless it was a catch-all "error", which is retried
    (transient network faults). delay throttles requests to respect YouTube.
    checkpoint controls how many classifications accumulate before an atomic
    write and optional flush. Rows are written in dataset order. Returns a
    Counter of status over all settled rows.
    """
    node_bin = node_bin or config.NODE_BIN
    by_key = {r["key"]: r for r in _load_rows(out_csv)}
    todo = [r for r in records
            if r["key"] not in by_key or by_key[r["key"]].get("status") == "error"]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"census: nothing to do ({len(by_key)} already classified)")
        return Counter(r["status"] for r in by_key.values())

    import yt_dlp
    opts = _build_opts(cookies, node_bin, video_url(todo[0]["key"]))
    n = len(todo)
    with yt_dlp.YoutubeDL(opts) as ydl:
        for i, rec in enumerate(todo, 1):
            status, detail = _classify(ydl, rec["key"])
            by_key[rec["key"]] = {
                "video_id": rec.get("video_id", ""),
                "key": rec["key"],
                "label": rec.get("label", ""),
                "status": status,
                "detail": detail,
            }
            print(f"[{i}/{n}] {rec['key']} -> {status}")
            if i % checkpoint == 0 or i == n:
                ordered = [by_key[r["key"]] for r in records if r["key"] in by_key]
                _atomic_write(out_csv, ordered)
                if flush:
                    flush(out_csv)
            if i < n:
                time.sleep(delay)
    return Counter(r["status"] for r in by_key.values())


def processable_keys(avail_csv, statuses=PROCESSABLE):
    """The video keys whose census verdict allows a download."""
    return {r["key"] for r in _load_rows(avail_csv)
            if r.get("status") in statuses}


def save_processable(links_csv, avail_csv, out_csv, statuses=PROCESSABLE,
                     flush=None):
    """Write the downloadable subset of the dataset CSV, in dataset order.

    The output keeps the original columns, so dvlog.load_records reads it
    unchanged. An extra availability column records the census verdict. The
    write is atomic and optionally Drive-flushed. Returns (count, label Counter).
    """
    status_by_key = {r["key"]: r["status"] for r in _load_rows(avail_csv)}
    keep = {k for k, s in status_by_key.items() if s in statuses}
    with open(links_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = []
        for r in reader:
            key = (r.get("key") or "").strip()
            if key in keep:
                r["availability"] = status_by_key[key]
                rows.append(r)
    _atomic_write(out_csv, rows, fields + ["availability"])
    if flush:
        flush(out_csv)
    counts = Counter((r.get("label") or "").strip() for r in rows)
    print(f"processable: {len(rows)} of {len(status_by_key)} classified")
    for label, k in sorted(counts.items()):
        print(f"  {label}: {k}")
    print(f"out: {out_csv}")
    return len(rows), counts


def _census_cli(args):
    records = dvlog.load_records(args.links)
    print(f"census: {len(records)} links from {args.links}")
    flush = _drive_flush if args.drive_flush else None
    counts = census(records, args.cookies, args.out, node_bin=args.node_bin,
                    delay=args.delay, limit=args.limit, flush=flush,
                    checkpoint=args.checkpoint)
    total = sum(counts.values())
    print(f"FINAL: {total} classified")
    for status, k in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {k}")
    print(f"out: {args.out}")
    return counts


def _processable_cli(args):
    flush = _drive_flush if args.drive_flush else None
    return save_processable(args.links, args.avail, args.out, flush=flush)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="D-Vlog availability census and processable selection")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="classify each video public/private")
    c.add_argument("--links", default=str(config.DVLOG_CSV),
                   help="dataset CSV with the video links")
    c.add_argument("--cookies", default=str(config.YT_COOKIES),
                   help="account cookie file (values are never printed)")
    c.add_argument("--out", required=True, help="result CSV path on Drive")
    c.add_argument("--node-bin", default=config.NODE_BIN)
    c.add_argument("--delay", type=float, default=1.0,
                   help="seconds between requests")
    c.add_argument("--limit", type=int, default=None,
                   help="classify at most this many new videos")
    c.add_argument("--checkpoint", type=int, default=25,
                   help="write and flush every this many videos")
    c.add_argument("--drive-flush", action="store_true",
                   help="flush Drive at each checkpoint (Colab only)")
    c.set_defaults(func=_census_cli)

    s = sub.add_parser("processable", help="save the downloadable subset")
    s.add_argument("--links", default=str(config.DVLOG_CSV),
                   help="original dataset CSV (raw columns are preserved)")
    s.add_argument("--avail", required=True, help="census result CSV")
    s.add_argument("--out", required=True, help="filtered dataset CSV path")
    s.add_argument("--drive-flush", action="store_true",
                   help="flush Drive after the write (Colab only)")
    s.set_defaults(func=_processable_cli)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()