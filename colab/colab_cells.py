"""Colab cell sources for the D-Vlog canary-then-full run.

Plain cell texts. Inject via the Colab MCP (append-only) or paste into a T4
GPU notebook in order. The zip `h5-source.zip` must sit in Drive at
DAIC-WOZ_Datasets/ before BOOTSTRAP runs.

Canary order: BOOTSTRAP -> DEPS -> BUILD -> CANARY_RUN -> VERIFY.
If VERIFY passes, run FULL (resumes from the same Drive manifest).
Each cell is bounded so a single MCP run_code_cell call does not block too long.
"""

BOOTSTRAP = r'''
# mount Drive and unpack the source zip
from google.colab import drive
drive.mount("/content/drive")
import zipfile, shutil, sys
from pathlib import Path
DRIVE = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
repo = Path("/content/useless")
zip_path = DRIVE / "h5-source.zip"
assert zip_path.exists(), f"upload h5-source.zip to {DRIVE} first"
if repo.exists():
    shutil.rmtree(repo)
repo.mkdir(parents=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(repo)
sys.path.insert(0, str(repo))
print("source extracted to", repo)
print("csv present:", (repo / "dvlog-video-links.csv").exists())
'''

DEPS = r'''
# install python deps + ffmpeg
%cd /content/useless
!python colab/dvlog_job.py --stage deps --repo /content/useless
'''

BUILD = r'''
# build the Rust feature CLI (release)
%cd /content/useless
!python colab/dvlog_job.py --stage build --repo /content/useless
'''

CANARY_RUN = r'''
# process ONE video end to end, IN THE KERNEL so the Drive flush works.
# A !python subprocess cannot flush Drive; a reset would lose the batch.
import sys
from pathlib import Path
repo = Path("/content/useless")
for m in list(sys.modules):
    if m == "preprocessing" or m.startswith("preprocessing.") or m in ("run", "dvlog_job"):
        del sys.modules[m]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "colab"))
import run as colab_run
colab_run.main(["--mode", "canary", "--repo", str(repo)])
'''

VERIFY = r'''
# verify the canary outputs on Drive
import csv, json, datetime
from pathlib import Path
from collections import Counter
DRIVE = Path("/content/drive/MyDrive/DAIC-WOZ_Datasets")
stamp = datetime.date.today().isoformat()
out = DRIVE / "features" / "D-Vlog" / stamp
wide = out / "features_wide.csv"
rel = out / "reliability.csv"
man = out / "manifest.json"
print("out root:", out, "exists:", out.exists())
assert wide.exists(), "canary produced no features_wide.csv"
with open(wide, newline="") as f:
    rows = list(csv.DictReader(f))
ncols = len(rows[0]) if rows else 0
print("rows:", len(rows), "columns:", ncols, "(expect 254)")
cnt = Counter()
with open(rel, newline="") as f:
    r = csv.reader(f); next(r, None)
    for row in r:
        cnt[row[2]] += 1
print("reliability counts:", dict(cnt))
emb = list((out / "batches").glob("batch_*/embeddings/*.json"))
print("embedding files:", len(emb))
if emb:
    e = json.loads(emb[0].read_text())
    print("modalities:", list(e), "dims:", {k: len(v) for k, v in e.items()})
print("manifest:", json.loads(man.read_text()) if man.exists() else "none")
ok = bool(rows) and ncols == 254 and bool(emb)
print("CANARY OK" if ok else "CANARY INCOMPLETE")
'''

FULL = r'''
# process ALL videos (resumes; skips finished videos). IN THE KERNEL so each
# batch flushes to Drive cloud before the next batch starts.
import sys
from pathlib import Path
repo = Path("/content/useless")
for m in list(sys.modules):
    if m == "preprocessing" or m.startswith("preprocessing.") or m in ("run", "dvlog_job"):
        del sys.modules[m]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "colab"))
import run as colab_run
colab_run.main(["--mode", "full", "--repo", str(repo)])
'''

# (name, source, per-cell run timeout seconds)
CANARY_CELLS = [
    ("bootstrap", BOOTSTRAP, 180),
    ("deps", DEPS, 600),
    ("build", BUILD, 600),
    ("canary_run", CANARY_RUN, 1500),
    ("verify", VERIFY, 180),
]

FULL_CELLS = [
    ("full", FULL, 7200),
]