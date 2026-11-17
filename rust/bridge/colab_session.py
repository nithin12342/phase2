"""Persistent Colab MCP session (stdlib only).

Keeps ONE server process alive for a whole job:
  1. initialize
  2. open_colab_browser_connection  (the user approves in the browser)
  3. dump the unlocked notebook tool schemas to colab_tools.json
  4. optionally execute a script of steps in the SAME session

Usage:
  python rust/bridge/colab_session.py --discover
  python rust/bridge/colab_session.py --script job.json --out results.json

Script steps (JSON list):
  {"tool": NAME, "arguments": {...}, "timeout": SECONDS}
  {"add_and_run": {"code": "...", "language": "python"}, "timeout": SECONDS}

The add_and_run step appends a code cell at the end of the notebook, reads the
returned cell id, then runs it. It only ADDS and RUNS cells; it never deletes,
moves, or updates existing cells (append-only rule). Tool names are dynamic;
read them from colab_tools.json after a --discover run.
"""
import argparse
import json
import sys
from pathlib import Path

from colab_mcp_driver import McpStdioClient


def connect(client, connect_timeout):
    client.initialize()
    print("opening browser connection (approve it in the browser)...", flush=True)
    result = client.call_tool("open_colab_browser_connection", {},
                              timeout=connect_timeout)
    print("connect result:", json.dumps(result), flush=True)
    return result


def dump_tools(client, path):
    tools = client.list_tools()
    Path(path).write_text(json.dumps(tools, indent=2), encoding="utf-8")
    print(f"tools ({len(tools)}) -> {path}", flush=True)
    for t in tools:
        print(f"  - {t['name']}", flush=True)
    return tools


def payload(res):
    """Extract the useful payload from an MCP tool result."""
    if isinstance(res, dict):
        if "structuredContent" in res:
            sc = res["structuredContent"]
            if isinstance(sc, dict) and "result" in sc:
                return sc["result"]
            return sc
        content = res.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("text", "")
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
    return res


def cell_id(res):
    p = payload(res)
    if isinstance(p, str):
        return p.strip()
    if isinstance(p, dict):
        for k in ("newCellId", "cellId", "id", "result"):
            if isinstance(p.get(k), str):
                return p[k]
    raise RuntimeError("cannot find cell id in: " + json.dumps(res)[:600])


def count_cells(client):
    res = client.call_tool("get_cells", {})
    p = payload(res)
    if isinstance(p, list):
        return len(p)
    if isinstance(p, dict):
        for k in ("cells", "result"):
            if isinstance(p.get(k), list):
                return len(p[k])
    print("WARN get_cells shape:", json.dumps(res)[:600], flush=True)
    return 0


def add_and_run(client, code, language, index, timeout):
    add_res = client.call_tool(
        "add_code_cell",
        {"cellIndex": index, "language": language, "code": code})
    cid = cell_id(add_res)
    print(f"  added cell {cid} at index {index}", flush=True)
    run_res = client.call_tool("run_code_cell", {"cellId": cid}, timeout=timeout)
    return {"cellId": cid, "index": index, "add": add_res, "run": run_res}


def run_script(client, script_path, out_path):
    steps = json.loads(Path(script_path).read_text(encoding="utf-8"))
    next_index = count_cells(client)
    print(f"notebook has {next_index} cells; appending new cells", flush=True)
    results = []
    for i, step in enumerate(steps):
        timeout = step.get("timeout", 600.0)
        try:
            if "add_and_run" in step:
                spec = step["add_and_run"]
                name = spec.get("name", f"cell_{i}")
                print(f"[{i + 1}/{len(steps)}] add+run {name}", flush=True)
                out = add_and_run(client, spec["code"],
                                  spec.get("language", "python"),
                                  next_index, timeout)
                next_index += 1
                results.append({"step": i, "name": name, "ok": True, **out})
                print("  output:", json.dumps(payload(out["run"]))[:3000],
                      flush=True)
            else:
                name = step["tool"]
                print(f"[{i + 1}/{len(steps)}] call {name}", flush=True)
                res = client.call_tool(name, step.get("arguments", {}),
                                       timeout=timeout)
                results.append({"step": i, "tool": name, "ok": True,
                                "result": res})
                print("  output:", json.dumps(payload(res))[:3000], flush=True)
        except Exception as e:  # noqa: BLE001 - report and stop the job
            results.append({"step": i, "ok": False, "error": str(e)})
            print(f"STEP FAILED: {e}", flush=True)
            break
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results -> {out_path}", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--connect-timeout", type=float, default=300.0)
    p.add_argument("--discover", action="store_true")
    p.add_argument("--script", default=None)
    p.add_argument("--tools-out", default="colab_tools.json")
    p.add_argument("--out", default="colab_results.json")
    args = p.parse_args(argv)

    client = McpStdioClient(timeout=args.timeout)
    client.start()
    try:
        connect(client, args.connect_timeout)
        dump_tools(client, args.tools_out)
        if args.script:
            run_script(client, args.script, args.out)
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())