"""Estimate each processable video's download size with a metadata-only request.

The program reads the processable dataset CSV and asks YouTube for each video's
format metadata, never the media bytes. It records the progressive format-18
size, which is the download target. The result is a resumable CSV: each
checkpoint writes atomically (temp file -> rename), so a stop never loses work.

Run it from a trusted (residential) IP. Public videos need no cookies there.
Pass --cookies only when a video requires authentication. Cookie values are
credentials; this module never prints them.
"""
import argparse
import csv
import shutil
import time
from collections import Counter
from pathlib import Path

from . import config
from .media import video_url

HEADER = ["video_id", "key", "label", "availability", "format_id",
          "size_bytes", "size_mb", "duration_s", "title", "status"]


def _clean(text, width=120):
    return " ".join(str(text).split())[:width]


def _load_links(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("key")]


def _load_rows(out_csv):
    out_csv = Path(out_csv)
    if not out_csv.exists():
        return []
    with open(out_csv, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("key")]


def _build_opts(cookies, node_bin, sample_url):
    import yt_dlp  # lazy: optional locally
    # Match the download (media.py): the mweb client serves progressive
    # format 18 (360p), the real download target. The default web client
    # serves only DASH, whose best size is 1080p and overstates the download.
    # mweb needs the node EJS solver to return formats without a bot check.
    node_bin = node_bin or shutil.which("node")
    argv = ["--extractor-args", "youtube:player_client=mweb",
            "--format", "18/b", "--no-warnings", sample_url]
    if node_bin:
        argv = ["--js-runtimes", f"node:{node_bin}",
                "--remote-components", "ejs:github"] + argv
    if cookies:
        argv = ["--cookies", str(cookies)] + argv
    parsed = yt_dlp.parse_options(argv)
    opts = next(x for x in parsed if isinstance(x, dict) and "format" in x)
    opts["skip_download"] = True
    opts["ignoreerrors"] = False
    opts["quiet"] = True
    opts["noprogress"] = True
    return opts


def _pick_format(info):
    fmts = info.get("formats") or []
    f18 = next((f for f in fmts if f.get("format_id") == "18"), None)
    if f18:
        return f18
    if info.get("filesize") or info.get("filesize_approx"):
        return info
    return fmts[-1] if fmts else {}


def _probe(ydl, rec):
    import yt_dlp
    base = {"video_id": rec.get("video_id", ""), "key": rec["key"],
            "label": rec.get("label", ""),
            "availability": rec.get("availability", "")}
    try:
        info = ydl.extract_info(video_url(rec["key"]), download=False)
    except yt_dlp.utils.DownloadError as err:
        msg = str(err).lower()
        status = "bot" if "not a bot" in msg else (
            "private" if "private" in msg else "error")
        return {**base, "status": status, "title": _clean(err)}
    except Exception as err:
        return {**base, "status": "error", "title": _clean(err)}
    if not info:
        return {**base, "status": "error", "title": "no metadata"}
    fmt = _pick_format(info)
    size = fmt.get("filesize") or fmt.get("filesize_approx") or 0
    return {**base, "format_id": fmt.get("format_id", ""),
            "size_bytes": size, "size_mb": round(size / 1e6, 2),
            "duration_s": info.get("duration") or 0,
            "title": _clean(info.get("title") or ""), "status": "ok"}


def _atomic_write(out_csv, rows):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})
    tmp.replace(out_csv)


def _summary(by_key):
    rows = list(by_key.values())
    ok = [r for r in rows if r.get("status") == "ok"]
    total = sum(int(r.get("size_bytes") or 0) for r in ok)
    return {"probed": len(rows), "ok": len(ok),
            "counts": dict(Counter(r.get("status") for r in rows)),
            "total_bytes": total, "total_gb": round(total / 1e9, 2)}


def probe_sizes(records, out_csv, cookies=None, node_bin=None, delay=1.0,
                limit=None, checkpoint=25):
    """Probe each unsettled video's size. Resume-safe and atomic.

    A prior row is kept unless its status was a catch-all "error", which is
    retried. delay throttles requests to respect YouTube. Rows are written in
    dataset order. Returns a summary dict with the total size.
    """
    by_key = {r["key"]: r for r in _load_rows(out_csv)}
    todo = [r for r in records
            if r["key"] not in by_key or by_key[r["key"]].get("status") == "error"]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"sizes: nothing to do ({len(by_key)} already probed)")
        return _summary(by_key)
    import yt_dlp
    opts = _build_opts(cookies, node_bin, video_url(todo[0]["key"]))
    n = len(todo)
    with yt_dlp.YoutubeDL(opts) as ydl:
        for i, rec in enumerate(todo, 1):
            row = _probe(ydl, rec)
            by_key[rec["key"]] = row
            print(f"[{i}/{n}] {rec['key']} -> {row['status']} "
                  f"{row.get('size_mb', '')} MB", flush=True)
            if i % checkpoint == 0 or i == n:
                ordered = [by_key[r["key"]] for r in records if r["key"] in by_key]
                _atomic_write(out_csv, ordered)
            if i < n:
                time.sleep(delay)
    return _summary(by_key)


def main(argv=None):
    p = argparse.ArgumentParser(description="D-Vlog processable video size probe")
    p.add_argument("--links", default=str(config.REPO_ROOT / "dvlog-processable.csv"),
                   help="processable dataset CSV")
    p.add_argument("--out", required=True, help="result size CSV path")
    p.add_argument("--cookies", default=None,
                   help="optional cookie file (values are never printed)")
    p.add_argument("--node-bin", default=None,
                   help="optional node path for the EJS solver")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between requests")
    p.add_argument("--limit", type=int, default=None,
                   help="probe at most this many new videos")
    p.add_argument("--checkpoint", type=int, default=25,
                   help="write every this many videos")
    p.add_argument("--public-only", action="store_true",
                   help="probe only public videos (skip unlisted)")
    args = p.parse_args(argv)

    if not (args.node_bin or shutil.which("node")):
        print("warning: node not found. The mweb client needs the node EJS "
              "solver to serve format 18. Install node or pass --node-bin.")
    records = _load_links(args.links)
    if args.public_only:
        records = [r for r in records if r.get("availability") == "public"]
    print(f"sizes: {len(records)} links from {args.links}")
    summary = probe_sizes(records, args.out, cookies=args.cookies,
                          node_bin=args.node_bin, delay=args.delay,
                          limit=args.limit, checkpoint=args.checkpoint)
    print("SUMMARY:", summary)
    print(f"out: {args.out}")
    return summary


if __name__ == "__main__":
    main()