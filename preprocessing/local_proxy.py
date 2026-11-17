"""A local forward proxy so Colab can download through this PC's trusted IP.

YouTube bot-flags the Colab datacenter IP but trusts this residential IP. Run
this proxy here, expose it to Colab with a TCP tunnel (bore), and pass the
tunnel URL to yt-dlp as --proxy. YouTube then sees the residential IP while the
video bytes save on Colab.

This is a forward proxy, not a MITM: it opens a raw TCP tunnel (HTTP CONNECT)
and shuttles bytes, so TLS stays end-to-end between yt-dlp and YouTube. It
binds 127.0.0.1 only; the tunnel, not the LAN, reaches it.

Safety: a public tunnel must use --auth (a shared secret) so strangers cannot
use it as an open proxy, and the destination allowlist limits it to YouTube /
Google video hosts. yt-dlp passes the secret as http://user:pass@host:port.

  python -m preprocessing.local_proxy --port 8888 --auth colab:SECRET
"""
import argparse
import base64
import select
import socket
import socketserver


# YouTube download infrastructure. The allowlist stops the proxy being used to
# reach arbitrary hosts if the tunnel endpoint is discovered.
ALLOW_SUFFIXES = (
    "youtube.com", "googlevideo.com", "googleapis.com", "ytimg.com",
    "ggpht.com", "google.com", "gstatic.com", "youtube-nocookie.com",
    # The yt-dlp EJS PO-token solver downloads from GitHub on first use.
    "github.com", "githubusercontent.com", "github.io",
)


def _allowed(host):
    host = host.lower().rstrip(".")
    return any(host == s or host.endswith("." + s) for s in ALLOW_SUFFIXES)


def _read_head(sock):
    """Read up to the end of the HTTP headers. Returns (head, leftover)."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > 65536:
            return None, b""
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def _headers(head):
    out = {}
    for line in head.split(b"\r\n")[1:]:
        if b":" in line:
            k, _, v = line.partition(b":")
            out[k.decode("latin1").strip().lower()] = v.decode("latin1").strip()
    return out


def _shuttle(a, b):
    """Copy bytes both ways until either side closes or errors."""
    socks = [a, b]
    try:
        while True:
            readable, _, errored = select.select(socks, [], socks, 120)
            if errored or not readable:
                return
            for s in readable:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        return


def _close(*socks):
    for s in socks:
        if s is None:
            continue
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        client = self.request
        client.settimeout(120)
        head, rest = _read_head(client)
        if not head:
            return
        request_line = head.split(b"\r\n")[0].decode("latin1")
        parts = request_line.split(" ")
        if len(parts) < 2:
            return
        method, target = parts[0].upper(), parts[1]
        headers = _headers(head)
        if self.server.auth and not self._auth_ok(headers):
            client.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\n"
                           b"Proxy-Authenticate: Basic realm=\"proxy\"\r\n\r\n")
            return
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
            self._connect(client, host, port, rest)
        else:
            self._plain(client, method, target, head, rest)

    def _auth_ok(self, headers):
        user, password = self.server.auth
        expect = "Basic " + base64.b64encode(
            f"{user}:{password}".encode()).decode()
        return headers.get("proxy-authorization") == expect

    def _connect(self, client, host, port, rest):
        if not _allowed(host):
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            remote = socket.create_connection((host, int(port)), timeout=30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if rest:
            remote.sendall(rest)
        try:
            _shuttle(client, remote)
        finally:
            _close(client, remote)

    def _plain(self, client, method, target, head, rest):
        try:
            without_scheme = target.split("://", 1)[1]
            hostport, _, path = without_scheme.partition("/")
            host, _, port = hostport.partition(":")
        except (IndexError, ValueError):
            client.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        if not _allowed(host):
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            remote = socket.create_connection((host, int(port or 80)), timeout=30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        request_line = f"{method} /{path} HTTP/1.1\r\n".encode("latin1")
        remote.sendall(request_line + head.split(b"\r\n", 1)[1] + b"\r\n\r\n" + rest)
        try:
            _shuttle(client, remote)
        finally:
            _close(client, remote)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    auth = None


def serve(host="127.0.0.1", port=8888, auth=None):
    server = _Server((host, port), _Handler)
    server.auth = auth
    print(f"proxy listening on {host}:{port} auth={'on' if auth else 'off'}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(description="local forward proxy for Colab")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--auth", default=None,
                   help="shared secret as user:pass; required for a public tunnel")
    args = p.parse_args(argv)
    auth = None
    if args.auth:
        user, _, password = args.auth.partition(":")
        auth = (user, password)
    serve(args.host, args.port, auth=auth)


if __name__ == "__main__":
    main()