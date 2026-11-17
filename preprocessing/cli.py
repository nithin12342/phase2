"""Command line for the D-Vlog binary-depression preprocessing pipeline.

  python -m preprocessing.cli --mode sample            # local, no GPU/network
  python -m preprocessing.cli --mode dry-run           # counts only
  python -m preprocessing.cli --mode run --out <drive> # Colab full processing

The sample mode is the local verification path. It uses synthetic signals and
the real Rust feature bridge, so it proves the wiring without any download.
"""
import argparse
import datetime
import json
from pathlib import Path

from . import config, dvlog, pipeline


def _default_out(mode):
    if mode == "run":
        stamp = datetime.date.today().isoformat()
        return config.DRIVE_BASE / config.FEATURES_SUBDIR / stamp
    return config.REPO_ROOT / ".local_out" / "dvlog_sample"


def build_parser():
    p = argparse.ArgumentParser(prog="preprocessing.cli", description=__doc__)
    p.add_argument("--mode", choices=["sample", "dry-run", "run"], default="sample")
    p.add_argument("--csv", default=str(config.DVLOG_CSV))
    p.add_argument("--out", default=None)
    p.add_argument("--raw", default=None, help="raw media dir (run mode)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = dvlog.load_records(args.csv)
    counts = dvlog.label_counts(records)
    out = Path(args.out) if args.out else _default_out(args.mode)

    if args.mode == "dry-run":
        summary = {
            "mode": "dry-run", "records": len(records),
            "label_counts": counts, "out_root": str(out),
            "batch_size": args.batch_size,
            "limit": args.limit if args.limit is not None else len(records),
        }
        print(json.dumps(summary, indent=2))
        return summary

    limit = args.limit
    if args.mode == "sample" and limit is None:
        limit = 4
    summary = pipeline.run(
        records, out, mode=args.mode, batch_size=args.batch_size,
        limit=limit, raw_root=args.raw)
    summary["label_counts"] = counts
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()