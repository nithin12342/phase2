"""All Colab orchestration logic lives here, not in notebook cells.

One reused notebook cell runs this file IN THE IPYTHON KERNEL (by import, not
with !python). The Drive flush needs the kernel; a !python subprocess cannot
flush, so a runtime reset would lose unflushed batches. The mode comes from
--mode or from the .colab_mode file written next to this script.

  import run; run.main(["--mode", "canary"])   # setup + 1 video + verify
  import run; run.main(["--mode", "full"])     # setup + all videos (resumable)
  import run; run.main(["--mode", "setup"])    # deps + rust build only
  import run; run.main(["--mode", "verify"])   # check the latest outputs only

Every batch streams to Drive, then flush_and_unmount blocks until the cloud
has it; a restart resumes safely and never loses a finished batch.
"""
import argparse
import datetime
import subprocess
import sys
import zipfile
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
DEFAULT_REPO = Path("/content/useless")


def mount_drive():
    # The upload cell mounts Drive inside the IPython kernel. A !python
    # subprocess has no kernel, so drive.mount fails there; skip when mounted.
    if Path("/content/drive/MyDrive").exists():
        return
    from google.colab import drive  # lazy: Colab only
    drive.mount("/content/drive")


def ensure_repo(repo):
    """Unpack the source zip from Drive when the repo is absent."""
    if (repo / "preprocessing").exists():
        return
    zp = DRIVE / "h5-source.zip"
    assert zp.exists(), f"missing source zip: {zp}"
    repo.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(repo)
    print("unpacked source to", repo, flush=True)


def read_mode_file(repo):
    mf = repo / "colab" / ".colab_mode"
    if mf.exists():
        return mf.read_text().strip()
    return None


def setup(repo):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "colab"))
    import dvlog_job
    dvlog_job.install_deps()
    dvlog_job.start_pot_server()
    dvlog_job.build_rust(str(repo))


def process(repo, limit, tag=None, links=None, proxy=None, out_root=None):
    sys.path.insert(0, str(repo))
    import time
    from preprocessing import config, dvlog, pipeline
    from dvlog_job import flush_drive
    if out_root is None:
        stamp = datetime.date.today().isoformat()
        if tag:
            stamp = f"{stamp}-{tag}"
        out_root = DRIVE / config.FEATURES_SUBDIR / stamp
    out_root = Path(out_root)
    raw_root = DRIVE / config.RAW_SUBDIR
    links = links or config.DVLOG_CSV
    records = dvlog.load_records(links)
    print("LINKS:", links, "records:", len(records),
          "labels:", dvlog.label_counts(records), flush=True)
    t0 = time.time()
    summary = pipeline.run(records, out_root, mode="run",
                           batch_size=config.BATCH_SIZE, limit=limit,
                           raw_root=raw_root, flush=flush_drive, proxy=proxy)
    elapsed = time.time() - t0
    summary["elapsed_s"] = round(elapsed, 1)
    summary["per_video_s"] = round(elapsed / max(1, summary["processed"]), 1)
    print("PIPELINE SUMMARY:", summary, flush=True)
    return out_root


def verify(out_root):
    import csv
    import json
    from collections import Counter
    wide = out_root / "features_wide.csv"
    rel = out_root / "reliability.csv"
    man = out_root / "manifest.json"
    print("out root:", out_root, "exists:", out_root.exists(), flush=True)
    if not wide.exists():
        print("CANARY INCOMPLETE (no features_wide.csv)", flush=True)
        return False
    with open(wide, newline="") as f:
        rows = list(csv.DictReader(f))
    ncols = len(rows[0]) if rows else 0
    cnt = Counter()
    with open(rel, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            cnt[row[2]] += 1
    emb = list((out_root / "batches").glob("batch_*/embeddings/*.json"))
    print("rows:", len(rows), "columns:", ncols, "(expect 254)", flush=True)
    print("reliability counts:", dict(cnt), flush=True)
    print("embedding files:", len(emb), flush=True)
    if emb:
        e = json.loads(emb[0].read_text())
        print("modalities:", list(e), "dims:", {k: len(v) for k, v in e.items()},
              flush=True)
    if man.exists():
        md = json.loads(man.read_text())
        print("manifest done:", len(md.get("done", {})),
              "failed:", len(md.get("failed", {})), flush=True)
    ok = bool(rows) and ncols == 254 and bool(emb)
    print("CANARY OK" if ok else "CANARY INCOMPLETE", flush=True)
    return ok


def latest_out_root():
    base = DRIVE / "features" / "D-Vlog"
    if not base.exists():
        return None
    stamps = sorted(p for p in base.iterdir() if p.is_dir())
    return stamps[-1] if stamps else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default=None,
                   choices=["setup", "canary", "full", "verify"])
    p.add_argument("--repo", default=str(DEFAULT_REPO))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", default=None,
                   help="suffix the dated output folder for a fresh run")
    p.add_argument("--links", default=None,
                   help="dataset CSV (default config.DVLOG_CSV; pass the "
                        "processable subset to skip unreachable videos)")
    p.add_argument("--proxy", default=None,
                   help="forward-proxy URL (tunnel to a trusted residential IP) "
                        "so YouTube sees that IP instead of the Colab IP")
    p.add_argument("--out-root", default=None,
                   help="pin the output folder (stable across days for a long "
                        "full run; default is a new dated subfolder)")
    args = p.parse_args(argv)
    repo = Path(args.repo)

    mount_drive()
    ensure_repo(repo)
    mode = args.mode or read_mode_file(repo) or "canary"
    print("MODE:", mode, "REPO:", repo, flush=True)

    if mode == "verify":
        out = latest_out_root()
        verify(out) if out else print("no outputs yet", flush=True)
        return
    if mode == "setup":
        setup(repo)
        return
    if mode == "canary":
        setup(repo)
        out = process(repo, limit=1 if args.limit is None else args.limit,
                      tag=args.tag, links=args.links, proxy=args.proxy,
                      out_root=args.out_root)
        verify(out)
        return
    if mode == "full":
        setup(repo)
        out = process(repo, limit=args.limit, tag=args.tag, links=args.links,
                      proxy=args.proxy, out_root=args.out_root)
        verify(out)
        return


if __name__ == "__main__":
    main()