"""Persistent Colab MCP daemon: connect ONCE, reuse the connection.

A new open_colab_browser_connection gives a fresh empty notebook, so this
daemon keeps ONE server process alive and reads commands from a local queue.
All orchestration logic lives in colab/run.py (uploaded code), not in cells.

Launch (background):
  python rust/bridge/colab_daemon.py \
      --cmd-dir .local_out/colab_cmd --out-dir .local_out/colab_out

Then drop command JSON files into --cmd-dir (atomic write). The daemon writes
each result to --out-dir/<same name>.json and removes the command file.

Commands:
  {"action": "upload", "mode": "canary"}   # push local zip -> session, set mode
  {"action": "run", "timeout": 1800}        # re-run the ONE reused run cell
  {"action": "eval", "code": "..."}         # add+run an arbitrary cell
  {"action": "cells"}                        # dump cells + outputs
  {"action": "quit"}                         # stop the daemon

Only ADD and RUN cells (plus reuse one run cell). Never delete/move existing
cells. Abort if the browser connect returns false.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

from colab_mcp_driver import McpStdioClient
from colab_session import payload, cell_id


def connect_checked(client, timeout):
    client.initialize()
    print("opening browser connection (approve ONCE in the browser)...",
          flush=True)
    res = client.call_tool("open_colab_browser_connection", {}, timeout=timeout)
    ok = payload(res) is True
    print("connect result:", json.dumps(res), flush=True)
    if not ok:
        raise RuntimeError("browser connect returned false; tools not unlocked")
    return res


def upload_cell_code(b64, mode, repo):
    return (
        "import base64, zipfile, shutil\n"
        "from pathlib import Path\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "DRIVE = Path('/content/drive/MyDrive/DAIC-WOZ_Datasets')\n"
        "DRIVE.mkdir(parents=True, exist_ok=True)\n"
        "repo = Path('" + str(repo) + "')\n"
        "mf = repo / 'colab' / '.colab_mode'\n"
        "old_mode = mf.read_text().strip() if mf.exists() else ''\n"
        "data = base64.b64decode('" + b64 + "')\n"
        "zp = DRIVE / 'h5-source.zip'\n"
        "tmp = zp.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(zp)\n"
        "if repo.exists(): shutil.rmtree(repo)\n"
        "repo.mkdir(parents=True)\n"
        "with zipfile.ZipFile(zp) as z:\n"
        "    bad = z.testzip(); z.extractall(repo)\n"
        "new_mode = '" + (mode or "") + "' or old_mode\n"
        "if new_mode:\n"
        "    (repo / 'colab' / '.colab_mode').write_text(new_mode)\n"
        "print('uploaded', len(data), 'bytes; corrupt:', bad, '; mode:', new_mode)\n"
    )


class Daemon:
    def __init__(self, client, zip_path, repo, out_dir):
        self.client = client
        self.zip_path = Path(zip_path)
        self.repo = repo
        self.out_dir = Path(out_dir)
        self.run_cell_id = None
        self.next_index = 0

    def _append_index(self):
        res = self.client.call_tool("get_cells", {})
        p = payload(res)
        n = len(p) if isinstance(p, list) else 0
        self.next_index = max(self.next_index, n)

    def add_run(self, code, timeout):
        self._append_index()
        add = self.client.call_tool("add_code_cell", {
            "cellIndex": self.next_index, "language": "python", "code": code})
        cid = cell_id(add)
        self.next_index += 1
        run = self.client.call_tool("run_code_cell", {"cellId": cid},
                                    timeout=timeout)
        return {"cellId": cid, "output": payload(run)}

    def ensure_run_cell(self):
        if self.run_cell_id is None:
            self._append_index()
            # Run in the IPython kernel, not !python: the Drive flush needs the
            # kernel. A subprocess cannot flush, so a reset would lose batches.
            code = (
                "import sys\n"
                "from pathlib import Path\n"
                "repo = Path('" + str(self.repo) + "')\n"
                "for m in list(sys.modules):\n"
                "    if m == 'preprocessing' or m.startswith('preprocessing.') "
                "or m in ('run', 'dvlog_job'):\n"
                "        del sys.modules[m]\n"
                "sys.path.insert(0, str(repo))\n"
                "sys.path.insert(0, str(repo / 'colab'))\n"
                "import run as colab_run\n"
                "colab_run.main([])\n"
            )
            add = self.client.call_tool("add_code_cell", {
                "cellIndex": self.next_index, "language": "python", "code": code})
            self.run_cell_id = cell_id(add)
            self.next_index += 1
            print("created reused run cell:", self.run_cell_id, flush=True)
        return self.run_cell_id

    def handle(self, cmd):
        action = cmd.get("action")
        if action == "quit":
            return {"output": "quitting"}
        if action == "upload":
            b64 = base64.b64encode(self.zip_path.read_bytes()).decode("ascii")
            code = upload_cell_code(b64, cmd.get("mode", ""), self.repo)
            return self.add_run(code, cmd.get("timeout", 300))
        if action == "run":
            cid = self.ensure_run_cell()
            run = self.client.call_tool("run_code_cell", {"cellId": cid},
                                        timeout=cmd.get("timeout", 1800))
            return {"cellId": cid, "output": payload(run)}
        if action == "eval":
            return self.add_run(cmd["code"], cmd.get("timeout", 600))
        if action == "cells":
            res = self.client.call_tool("get_cells", {"includeOutputs": True})
            return {"output": payload(res)}
        raise ValueError(f"unknown action: {action}")

    def write_status(self, **kw):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "_status.json").write_text(json.dumps({
            "connected": True, "run_cell_id": self.run_cell_id,
            "next_index": self.next_index, **kw}, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cmd-dir", default=".local_out/colab_cmd")
    p.add_argument("--out-dir", default=".local_out/colab_out")
    p.add_argument("--zip", default=".local_out/h5-source.zip")
    p.add_argument("--repo", default="/content/useless")
    p.add_argument("--connect-timeout", type=float, default=300.0)
    p.add_argument("--poll", type=float, default=1.0)
    args = p.parse_args(argv)

    cmd_dir = Path(args.cmd_dir)
    out_dir = Path(args.out_dir)
    cmd_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = McpStdioClient(timeout=60.0)
    client.start()
    try:
        connect_checked(client, args.connect_timeout)
        tools = client.list_tools()
        print(f"connected; {len(tools)} tools unlocked", flush=True)
        daemon = Daemon(client, args.zip, args.repo, out_dir)
        daemon.write_status(tools=[t["name"] for t in tools])
        print("READY — waiting for commands in", cmd_dir, flush=True)
        while True:
            if client._proc.poll() is not None:
                print("MCP server exited; stopping", flush=True)
                break
            for cmd_file in sorted(cmd_dir.glob("*.json")):
                try:
                    cmd = json.loads(cmd_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # partial write; retry next poll
                print("CMD:", cmd_file.name, cmd.get("action"), flush=True)
                try:
                    result = daemon.handle(cmd)
                    out = {"ok": True, "action": cmd.get("action"), **result}
                except Exception as e:  # noqa: BLE001 - report, keep daemon alive
                    out = {"ok": False, "action": cmd.get("action"),
                           "error": str(e)}
                    print("CMD ERROR:", e, flush=True)
                (out_dir / cmd_file.name).write_text(
                    json.dumps(out, indent=2), encoding="utf-8")
                cmd_file.unlink()
                daemon.write_status(last=cmd_file.name)
                if cmd.get("action") == "quit":
                    print("quit received; stopping", flush=True)
                    return
            time.sleep(args.poll)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())