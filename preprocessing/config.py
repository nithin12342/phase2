"""Pipeline configuration: paths, constants, and the binary label map."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Colab Drive base. Override with --out for local runs.
DRIVE_BASE = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
RAW_SUBDIR = "RAW/D-Vlog"
FEATURES_SUBDIR = "features/D-Vlog"
# Frames are regenerable intermediates: keep them on local disk, not Drive
# FUSE, where thousands of small reads/writes per video stall the pipeline.
FRAMES_LOCAL = Path("/content/h5_frames")

# YouTube download on a datacenter IP needs three things together: account
# cookies (pass the player bot check), a bgutil PO-token server (gvs tokens),
# and the EJS remote challenge solver run on node. All three are required.
YT_COOKIES = DRIVE_BASE / "cookies-yt.txt"
NODE_BIN = "/tools/node/bin/node"
POT_PORT = 4416
POT_REPO = Path("/content/bgutil-provider")
POT_TAG = "2.0.2"

BATCH_SIZE = 8
TARGET_FPS = 8
AUDIO_SR = 16000

RUST_CLI = REPO_ROOT / "rust" / "target" / "release" / "h5-preprocess-cli"
RUST_CLI_DEBUG = REPO_ROOT / "rust" / "target" / "debug" / "h5-preprocess-cli"
SCHEMA_DOC = REPO_ROOT / "FINAL_200_FEATURE_SCHEMA.md"
DVLOG_CSV = REPO_ROOT / "dvlog-video-links.csv"

# Binary depression target only (no PHQ-8 regression).
LABEL_MAP = {"depression": 1, "normal": 0}