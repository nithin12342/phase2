"""Send one command to the persistent Colab daemon and print its result.

  python rust/bridge/colab_send.py '{"action": "upload", "mode": "canary"}'
  python rust/bridge/colab_send.py '{"action": "run", "timeout": 1800}' --wait 1900

Writes the command atomically into the daemon's cmd dir, waits for the matching
result file, prints it, and exits. Does not open a browser connection.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", help="JSON command string")
    p.add_argument("--cmd-dir", default=".local_out/colab_cmd")
    p.add_argument("--out-dir", default=".local_out/colab_out")
    p.add_argument("--wait", type=float, default=300.0)
    args = p.parse_args(argv)

    cmd = json.loads(args.command)
    cmd_dir = Path(args.cmd_dir)
    out_dir = Path(args.out_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    name = f"cmd_{uuid.uuid4().hex[:12]}.json"
    cmd_file = cmd_dir / name
    tmp = cmd_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(cmd), encoding="utf-8")
    tmp.replace(cmd_file)

    out_file = out_dir / name
    deadline = time.time() + args.wait
    while time.time() < deadline:
        if out_file.exists():
            print(out_file.read_text(encoding="utf-8"))
            return 0
        time.sleep(1.0)
    print(json.dumps({"ok": False, "error": f"timed out waiting for {name}"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())