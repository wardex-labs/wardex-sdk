"""What wardex adds to the host's own thread per HTTP request it does not keep — run by hand.

    uv run python sdks/python/tests/bench_host_overhead.py           # the full run to report
    uv run python sdks/python/tests/bench_host_overhead.py --quick   # a smoke run

Ordinary traffic, outside any span, under the default `capture_mode`: the requests an agent
process makes that are not model calls (an internal JSON API, a download, a cache). Each workload
runs before `init()` and after it, in this process, against a loopback server, and the number that
matters is the calling thread's CPU time per request (`time.thread_time`): the work wardex puts
on the host's thread, which a loaded machine distorts far less than wall time. Wall time is printed
beside it.

Workloads: a small JSON reply and a 1.2 MB one, each over plaintext HTTP/1.1 and over TLS, and a
non-HTTP echo (a Redis-like protocol) that the seams must classify and then leave alone.

Deliberately not a pytest test: timings on a shared box are noise for a gate. The budget printed at
the end is the claim the README makes; a FAIL line is the regression to look at.
"""

from __future__ import annotations

import http.client
import http.server
import json
import socket
import ssl
import statistics
import sys
import threading
import time
from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"

#: Added host-thread CPU per request, in microseconds, that a workload may cost with wardex
#: installed. Measured on an arm64 macOS machine at rest: about 35 µs for a small request (the
#: fixed cost of reading its head and pairing its reply), the same for a 1.2 MB one (a body wardex
#: does not keep is counted, not copied), and 2 µs per echo. Budgets carry headroom for noise; the
#: TLS rows read their body in 16 KB records, so they are the noisiest.
BUDGET_US = {
    "plain small": 50.0,
    "plain 1.2MB": 50.0,
    "tls small": 50.0,
    "tls 1.2MB": 100.0,
    "echo": 10.0,
}


def _body(rows: int) -> bytes:
    return json.dumps({"rows": [{"id": i, "name": "x" * 40} for i in range(rows)]}).encode()


def _serve(body: bytes, tls: bool) -> int:
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(_FIXTURES / "cert.pem", _FIXTURES / "key.pem")
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def _serve_echo() -> int:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def accept() -> None:
        while True:
            conn, _ = listener.accept()

            def echo(c: socket.socket = conn) -> None:
                while data := c.recv(65536):
                    c.sendall(data)

            threading.Thread(target=echo, daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return listener.getsockname()[1]


def _http_round(port: int, tls: bool, n: int) -> tuple[float, float]:
    if tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            "127.0.0.1", port, context=ctx
        )
    else:
        conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/internal/rows")  # connect and handshake outside the timed loop
    conn.getresponse().read()
    cpu, wall = time.thread_time(), time.perf_counter()
    for _ in range(n):
        conn.request("GET", "/internal/rows")
        conn.getresponse().read()
    cpu, wall = time.thread_time() - cpu, time.perf_counter() - wall
    conn.close()
    return cpu / n * 1e6, wall / n * 1e6


def _echo_round(port: int, n: int) -> tuple[float, float]:
    sock = socket.create_connection(("127.0.0.1", port))
    msg = b"*1\r\n$4\r\nPING\r\n"
    cpu, wall = time.thread_time(), time.perf_counter()
    for _ in range(n):
        sock.sendall(msg)
        sock.recv(100)
    cpu, wall = time.thread_time() - cpu, time.perf_counter() - wall
    sock.close()
    return cpu / n * 1e6, wall / n * 1e6


def _measure(ports: dict[str, int], n: int, rounds: int) -> dict[str, tuple[float, float]]:
    """Median over `rounds` of the per-request (CPU µs, wall µs) of each workload."""
    out: dict[str, tuple[float, float]] = {}
    for name, port in ports.items():
        samples = []
        for _ in range(rounds):
            if name == "echo":
                samples.append(_echo_round(port, n * 5))
            else:
                samples.append(_http_round(port, name.startswith("tls"), n))
        out[name] = (
            statistics.median(s[0] for s in samples),
            statistics.median(s[1] for s in samples),
        )
    return out


def main() -> int:
    quick = "--quick" in sys.argv
    n, rounds = (50, 3) if quick else (400, 7)
    small, large = _body(20), _body(20_000)
    ports = {
        "plain small": _serve(small, tls=False),
        "plain 1.2MB": _serve(large, tls=False),
        "tls small": _serve(small, tls=True),
        "tls 1.2MB": _serve(large, tls=True),
        "echo": _serve_echo(),
    }
    print(f"bodies: small {len(small)} B, large {len(large)} B; {n} requests x {rounds} rounds")

    _measure(ports, max(5, n // 10), 1)  # warm the interpreter and the servers
    off = _measure(ports, n, rounds)

    import wardex_sdk as wardex

    wardex.init(
        backend=wardex.BackendConfig(endpoint="http://127.0.0.1:9/v1/traces"),
        adapters=wardex.AdaptersConfig(enabled=()),
    )
    _measure(ports, max(5, n // 10), 1)
    on = _measure(ports, n, rounds)
    wardex.close()

    failed = False
    print(f"{'workload':<12} {'cpu off':>9} {'cpu on':>9} {'added':>8} {'budget':>8}  wall off/on")
    for name in ports:
        added = on[name][0] - off[name][0]
        verdict = "ok" if added <= BUDGET_US[name] else "FAIL"
        failed |= verdict == "FAIL"
        print(
            f"{name:<12} {off[name][0]:8.1f}µs {on[name][0]:8.1f}µs {added:7.1f}µs "
            f"{BUDGET_US[name]:7.1f}µs  {off[name][1]:.0f}/{on[name][1]:.0f}µs  {verdict}"
        )
    print("budget: added host-thread CPU per request, ordinary traffic outside any span")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
