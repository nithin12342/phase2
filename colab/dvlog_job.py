"""Colab-side full D-Vlog processing job (T4 GPU runtime).

This wrapper does the Colab-only setup, then calls the same pipeline CLI used
for the local sample. Every batch streams to Google Drive with a manifest, so a
disconnect resumes without loss.

  python colab/dvlog_job.py --stage dry-run        # print the plan, no effects
  python colab/dvlog_job.py --stage all --repo /content/useless

WARNING: confirm the YouTube terms of service and the D-Vlog license before any
bulk download. The download stage is the only stage that touches the network.
"""
import argparse
import datetime
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

DRIVE_BASE = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
PIP_DEPS = [
    "yt-dlp", "openai-whisper", "librosa",
    "opencv-python-headless", "transformers", "pillow",
    "bgutil-ytdlp-pot-provider==2.0.2",
]


def mount_drive():
    from google.colab import drive  # lazy: Colab only
    drive.mount("/content/drive")
    print("drive mounted:", DRIVE_BASE.exists())


def flush_drive(out_root):
    """Block until Drive uploads reach the cloud, then remount and verify.

    Colab's Drive FUSE buffers writes; a runtime reset can drop unflushed
    files. flush_and_unmount waits for the cloud upload. The remount reads from
    cloud with an empty cache, so the manifest check proves persistence.

    WARNING: this needs the IPython kernel. A !python subprocess cannot flush.
    Run colab/run.py by import inside a notebook cell, not with !python.
    """
    from google.colab import drive  # lazy: Colab kernel only
    drive.flush_and_unmount()
    drive.mount("/content/drive")
    man = Path(out_root) / "manifest.json"
    if not man.exists():
        raise RuntimeError(f"manifest lost after Drive remount: {man}")
    print("drive flushed + verified:", man.name, flush=True)


def install_deps():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP_DEPS],
                   check=True)
    if not shutil.which("ffmpeg"):
        subprocess.run(["apt-get", "-y", "-qq", "install", "ffmpeg"], check=True)
    print("deps installed")


def build_rust(repo):
    rust_dir = Path(repo) / "rust"
    if not shutil.which("cargo"):
        subprocess.run(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --profile minimal", shell=True, check=True)
        os.environ["PATH"] += os.pathsep + str(Path.home() / ".cargo" / "bin")
    subprocess.run(["cargo", "build", "--release"], cwd=rust_dir, check=True)
    print("rust built:", (rust_dir / "target/release/h5-preprocess-cli").exists())


def _pot_listening(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_pot_server():
    """Start the bgutil PO-token HTTP server. Build it first when absent.

    The server supplies gvs PO tokens for the stream URLs. Account cookies
    pass the player-stage bot check. A datacenter IP needs both together.
    A fresh VM rebuilds the server here, so a restart resumes safely.
    """
    from preprocessing import config  # repo is on sys.path by setup()
    port = config.POT_PORT
    if _pot_listening(port):
        print("pot server already running on", port, flush=True)
        return
    repo = config.POT_REPO
    server = repo / "server"
    main_js = server / "build" / "main.js"
    if not main_js.exists():
        subprocess.run(
            ["git", "clone", "--branch", config.POT_TAG, "--depth", "1",
             "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
             str(repo)], check=True)
        subprocess.run(["npm", "ci"], cwd=server, check=True)
        subprocess.run(["npx", "tsc"], cwd=server, check=True)
    log = open(repo / "pot.log", "ab")
    subprocess.Popen([config.NODE_BIN, str(main_js)],
                     stdout=log, stderr=log, cwd=server)
    for _ in range(30):
        if _pot_listening(port):
            print("pot server started on", port, flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"pot server did not start on port {port}")


def run_pipeline(repo, limit, batch_size):
    sys.path.insert(0, str(repo))
    from preprocessing import config, dvlog, pipeline  # noqa: E402

    stamp = datetime.date.today().isoformat()
    out_root = DRIVE_BASE / config.FEATURES_SUBDIR / stamp
    raw_root = DRIVE_BASE / config.RAW_SUBDIR
    records = dvlog.load_records(config.DVLOG_CSV)
    summary = pipeline.run(
        records, out_root, mode="run", batch_size=batch_size,
        limit=limit, raw_root=raw_root, flush=flush_drive)
    print(summary)
    return summary


def dry_run(repo, limit, batch_size):
    print({
        "repo": str(repo),
        "drive_base": str(DRIVE_BASE),
        "stages": ["mount", "deps", "build", "run"],
        "pip_deps": PIP_DEPS,
        "batch_size": batch_size,
        "limit": limit,
        "note": "run mode downloads YouTube videos; confirm license first",
    })


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="dry-run",
                   choices=["dry-run", "mount", "deps", "build", "run", "all"])
    p.add_argument("--repo", default=os.environ.get("H5_REPO", "/content/useless"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)

    if args.stage == "dry-run":
        return dry_run(args.repo, args.limit, args.batch_size)
    if args.stage in ("mount", "all"):
        mount_drive()
    if args.stage in ("deps", "all"):
        install_deps()
    if args.stage in ("build", "all"):
        build_rust(args.repo)
    if args.stage in ("run", "all"):
        run_pipeline(args.repo, args.limit, args.batch_size)


if __name__ == "__main__":
    main()