"""Programmatic Colab MCP driver (stdlib only).

Spawns `uvx git+https://github.com/googlecolab/colab-mcp` over stdio and speaks
JSON-RPC 2.0 directly. This does not rely on Qoder's native MCP support.

Subcommands:
  list     Spawn the server, initialize, list tools, print, exit (smoke test).
  connect  Call open_colab_browser_connection, wait for the browser, list tools.
  call     Call a named tool with a JSON arguments file.

The notebook tools are injected at runtime after the browser connects. Their
names are dynamic; always read them from `tools/list`. Do not hardcode them.
"""
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

LAUNCH = ["git+https://github.com/googlecolab/colab-mcp"]
PROTOCOL_VERSION = "2024-11-05"


class McpStdioClient:
    """A minimal MCP client over a subprocess stdio pipe."""

    def __init__(self, timeout=60.0):
        self.timeout = timeout
        self._id = 0
        self._inbox = queue.Queue()
        self._stderr = []
        self._proc = None
        self._stop = threading.Event()

    def start(self):
        uvx = shutil.which("uvx")
        if not uvx:
            raise RuntimeError("uvx not found on PATH")
        self._proc = subprocess.Popen(
            [uvx, *LAUNCH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=self._server_env(),
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    @staticmethod
    def _server_env():
        # Newer httpx (pulled by fastmcp's startup version check) crashes on
        # IPv6 tokens like "[::1]" in NO_PROXY when trust_env builds mounts.
        # No proxy is configured here, so drop IPv6 tokens for the server only.
        env = os.environ.copy()
        for key in ("NO_PROXY", "no_proxy"):
            if key in env:
                tokens = [t for t in env[key].split(",")
                          if ":" not in t and "[" not in t]
                env[key] = ",".join(tokens)
        return env

    def _read_stdout(self):
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self._inbox.put({"_eof": True})

    def _read_stderr(self):
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())

    def _send(self, obj):
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _next_id(self):
        self._id += 1
        return self._id

    def request(self, method, params=None, timeout=None):
        rid = self._next_id()
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            try:
                item = self._inbox.get(timeout=0.5)
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise RuntimeError("server exited: " + self.stderr_tail())
                continue
            if item.get("_eof"):
                raise RuntimeError("server closed stdout: " + self.stderr_tail())
            if item.get("id") == rid:
                if "error" in item:
                    raise RuntimeError(f"{method} error: {item['error']}")
                return item.get("result")
            # Notifications and other responses are ignored here.
        raise TimeoutError(f"{method} timed out: " + self.stderr_tail())

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def initialize(self):
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "h5-driver", "version": "0.1.0"},
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self):
        result = self.request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name, arguments=None, timeout=None):
        return self.request("tools/call", {
            "name": name, "arguments": arguments or {},
        }, timeout=timeout)

    def stderr_tail(self, n=8):
        return " | ".join(self._stderr[-n:])

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def print_tools(tools):
    print(f"tools ({len(tools)}):")
    for t in tools:
        desc = (t.get("description") or "").splitlines()[0:1]
        print(f"  - {t['name']}: {desc[0] if desc else ''}")


def cmd_list(args):
    client = McpStdioClient(timeout=args.timeout)
    client.start()
    try:
        info = client.initialize()
        server = info.get("serverInfo", {})
        print(f"server: {server.get('name')} {server.get('version')}")
        print(f"protocol: {info.get('protocolVersion')}")
        print_tools(client.list_tools())
    finally:
        client.close()


def cmd_connect(args):
    client = McpStdioClient(timeout=args.timeout)
    client.start()
    try:
        client.initialize()
        print("calling open_colab_browser_connection (connect in the browser)...")
        result = client.call_tool("open_colab_browser_connection", {},
                                  timeout=args.connect_timeout)
        print("connect result:", json.dumps(result))
        print_tools(client.list_tools())
    finally:
        client.close()


def cmd_call(args):
    with open(args.args_file, encoding="utf-8") as f:
        arguments = json.load(f)
    client = McpStdioClient(timeout=args.timeout)
    client.start()
    try:
        client.initialize()
        result = client.call_tool(args.tool, arguments, timeout=args.connect_timeout)
        print(json.dumps(result, indent=2))
    finally:
        client.close()


def main():
    p = argparse.ArgumentParser(description="Programmatic Colab MCP driver")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("connect")
    c = sub.add_parser("call")
    c.add_argument("tool")
    c.add_argument("args_file")
    args = p.parse_args()
    {"list": cmd_list, "connect": cmd_connect, "call": cmd_call}[args.cmd](args)


if __name__ == "__main__":
    main()