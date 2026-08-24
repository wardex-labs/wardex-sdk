"""Parse-off-loop benchmark — run by hand, never collected by pytest.

Three measurements (design §4.10), each printing a table row:

  1. asyncio EVENT-LOOP DRIFT under concurrent SSE streaming, wardex on vs
     off, h1 and h2, three body sizes. The claim under test is the README's
     "Capture itself never blocks your coroutines": wardex-attributed drift
     p99 <= 0.5 ms, and max drift INDEPENDENT of body size (<= 0.5 ms spread
     between 256 KB and 4 MB).
  2. sync THROUGHPUT: the request loop's wall clock must not regress (the
     parse left the hot path), and the flush()-inclusive total must stay
     within ~5% (same CPU, moved threads).
  3. GIL RELEASE: a competing pure-Python thread must keep >= 90% of its
     solo rate while 8 MB parses run on another thread.

Usage:
    uv run python sdks/python/tests/bench_parse_off_loop.py [--quick]

`--quick` shrinks rounds/requests for a smoke run; report REAL numbers from
the full run. Like `e2e_split_export_phoenix.py`, the filename misses
pytest's `python_files` on purpose. Run once more under `.venv-py310` after
`scripts/check-py310.sh` — nothing else executes this file on the floor.
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import json
import ssl
import sys
import threading
import time
from pathlib import Path

import httpx

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import counters
from wardex_sdk._config import BatchingConfig
from wardex_sdk.testing import RecordingTransport

_FIXTURES = Path(__file__).parent / "fixtures"


def _verify_ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(_FIXTURES / "cert.pem"))


def _server_ssl_ctx(alpn: list[str] | None = None) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(_FIXTURES / "cert.pem"), keyfile=str(_FIXTURES / "key.pem"))
    if alpn:
        ctx.set_alpn_protocols(alpn)
    return ctx


def _sse_body(target_bytes: int) -> bytes:
    word = "x" * 512
    chunks: list[bytes] = []
    size = 0
    i = 0
    while size < target_bytes:
        payload = {
            "id": "chatcmpl-bench",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": f"{word}{i} "}, "finish_reason": None}],
        }
        chunk = b"data: " + json.dumps(payload).encode() + b"\n\n"
        chunks.append(chunk)
        size += len(chunk)
        i += 1
    chunks.append(
        b'data: {"id":"chatcmpl-bench","object":"chat.completion.chunk",'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
    )
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# servers
# ---------------------------------------------------------------------------


class _H1SseServer:
    """Threaded HTTP/1.1 TLS server answering every POST with `body`."""

    def __init__(self, body: bytes) -> None:
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                for start in range(0, len(outer.body), 256 * 1024):
                    self.wfile.write(outer.body[start : start + 256 * 1024])
                    self.wfile.flush()

            def log_message(self, *args: object) -> None:
                pass

        self.body = body

        class _Server(http.server.ThreadingHTTPServer):
            request_queue_size = 128  # 32 concurrent connects overflow the default 5

        self._httpd = _Server(("127.0.0.1", 0), _Handler)
        self._httpd.socket = _server_ssl_ctx().wrap_socket(self._httpd.socket, server_side=True)
        host, port = self._httpd.socket.getsockname()[:2]
        self.url = f"https://{host}:{port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class _H2SseServer:
    """Minimal TLS h2 server (the `h2` package) — every stream gets `body`.

    One connection, many multiplexed streams: the fan-in case where N
    responses complete inside the same loop iteration, which is the one
    place the per-chunk-feed re-litigation trigger (p99 50 µs) could show.
    """

    def __init__(self, body: bytes) -> None:
        import socket as socketlib

        self.body = body
        ctx = _server_ssl_ctx(alpn=["h2"])
        sock = socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM)
        sock.setsockopt(socketlib.SOL_SOCKET, socketlib.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        self._tls = ctx.wrap_socket(sock, server_side=True)
        self._tls.settimeout(0.5)
        host, port = self._tls.getsockname()[:2]
        self.url = f"https://{host}:{port}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _handle(self, conn) -> None:
        import h2.config
        import h2.connection
        import h2.events

        config = h2.config.H2Configuration(client_side=False)
        h2conn = h2.connection.H2Connection(config=config)
        h2conn.initiate_connection()
        conn.sendall(h2conn.data_to_send())
        conn.settimeout(1.0)
        # stream id -> bytes already sent. Event-driven: every recv's events
        # are processed (a RequestReceived arriving mid-send must not be
        # dropped), and sending round-robins whatever the windows allow.
        pending: dict[int, int] = {}
        while not self._stop.is_set():
            for sid in list(pending):
                offset = pending[sid]
                while offset < len(self.body):
                    try:
                        window = min(
                            h2conn.local_flow_control_window(sid),
                            h2conn.max_outbound_frame_size,
                            len(self.body) - offset,
                        )
                    except Exception:
                        pending.pop(sid, None)
                        break
                    if window <= 0:
                        break
                    h2conn.send_data(sid, self.body[offset : offset + window])
                    offset += window
                    out = h2conn.data_to_send()
                    if out:
                        conn.sendall(out)
                if sid in pending:
                    pending[sid] = offset
                    if offset >= len(self.body):
                        try:
                            h2conn.end_stream(sid)
                        except Exception:
                            pass
                        pending.pop(sid, None)
            out = h2conn.data_to_send()
            if out:
                conn.sendall(out)
            try:
                data = conn.recv(65535)
            except TimeoutError:
                continue
            except OSError:
                return
            if not data:
                return
            for event in h2conn.receive_data(data):
                if isinstance(event, h2.events.RequestReceived):
                    h2conn.send_headers(
                        event.stream_id,
                        [(":status", "200"), ("content-type", "text/event-stream")],
                    )
                    pending[event.stream_id] = 0
            out = h2conn.data_to_send()
            if out:
                conn.sendall(out)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._tls.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=self._safe_handle, args=(client,), daemon=True).start()

    def _safe_handle(self, client) -> None:
        try:
            self._handle(client)
        except Exception:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._tls.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# measurement 1 — event-loop drift
# ---------------------------------------------------------------------------


async def _drift_round(url: str, concurrency: int, http2: bool) -> list[float]:
    drifts: list[float] = []
    stop = asyncio.Event()

    async def watchdog() -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            before = loop.time()
            await asyncio.sleep(0.001)
            drifts.append(loop.time() - before - 0.001)

    dog = asyncio.ensure_future(watchdog())
    async with httpx.AsyncClient(verify=_verify_ctx(), http2=http2, timeout=120.0) as client:
        await asyncio.gather(
            *(
                client.post(f"{url}/v1/chat/completions", json={"model": "gpt-4o-mini"})
                for _ in range(concurrency)
            )
        )
    await asyncio.sleep(0.05)  # catch a stall at completion fan-in
    stop.set()
    await dog
    return drifts


def _measure_drift(url: str, rounds: int, concurrency: int, http2: bool) -> dict[str, float]:
    drifts: list[float] = []
    for _ in range(rounds):
        drifts.extend(asyncio.run(_drift_round(url, concurrency, http2)))
    drifts.sort()
    return {
        "p50": drifts[len(drifts) // 2],
        "p99": drifts[int(len(drifts) * 0.99)],
        "max": drifts[-1],
    }


def bench_loop_drift(rounds: int, concurrency: int) -> None:
    print("\n== 1. asyncio event-loop drift (wardex-attributed = on - off) ==")
    print(f"   {concurrency} concurrent streams x {rounds} rounds, 1 ms watchdog")
    header = (
        f"{'variant':<10} {'size':>7} {'off p99':>9} {'on p99':>9} {'attr p99':>9} {'attr max':>9}"
    )
    print(header)
    print("-" * len(header))
    max_by_size: dict[tuple[str, int], float] = {}
    for http2 in (False, True):
        variant = "h2" if http2 else "h1"
        for size in (256 * 1024, 1024 * 1024, 4 * 1024 * 1024):
            body = _sse_body(size)
            print(f"   ... {variant} {size // 1024}K", flush=True)
            server: _H1SseServer | _H2SseServer = (
                _H2SseServer(body) if http2 else _H1SseServer(body)
            )
            try:
                off = _measure_drift(server.url, rounds, concurrency, http2)
                transport = RecordingTransport()
                # A short flush interval keeps the span buffer continuously
                # drained into the recording transport: without it, rounds of
                # multi-MB spans cross max_buffer_bytes and the drop-oldest
                # eviction (working as designed) reads as missing spans here.
                wardex.init(
                    intercept=True,
                    transport=transport,
                    batching=BatchingConfig(flush_interval=0.05),
                )
                counters.reset()
                try:
                    on = _measure_drift(server.url, rounds, concurrency, http2)
                    client = _hub.get_client()
                    backlog_peak = client._finalize.pending()
                    wardex.flush()
                    spans = [s for s in transport.spans]
                    expected = rounds * concurrency
                    evicted = counters.get("client.finalize.backlog_evicted")
                finally:
                    wardex.close()
                attr_p99 = max(0.0, on["p99"] - off["p99"])
                attr_max = max(0.0, on["max"] - off["max"])
                # The gate is defined on the ATTRIBUTED stall (on - off): the
                # off-baseline itself grows with body size, so storing the on
                # absolute here measured the machine plus wardex and the
                # spread line neither passed nor failed the stated target.
                max_by_size[(variant, size)] = attr_max
                print(
                    f"{variant:<10} {size // 1024:>6}K "
                    f"{1000 * off['p99']:>8.3f} {1000 * on['p99']:>8.3f} "
                    f"{1000 * attr_p99:>8.3f} {1000 * attr_max:>8.3f}   (ms)"
                )
                if len(spans) != expected:
                    print(
                        f"   !! span count {len(spans)} != requests {expected} "
                        "(capture regression — numbers above are void)"
                    )
                if evicted:
                    print(
                        f"   ({evicted} backlog eviction(s): {concurrency} x "
                        f"{size // 1024}K in flight crossed max_parse_backlog_bytes; "
                        "those spans shipped as caller-side fallbacks)"
                    )
                if backlog_peak > 0:
                    print(f"   (backlog still held {backlog_peak} at measure end)")
            finally:
                server.close()
    for variant in ("h1", "h2"):
        small = max_by_size.get((variant, 256 * 1024))
        big = max_by_size.get((variant, 4 * 1024 * 1024))
        if small is not None and big is not None:
            print(
                f"   {variant}: ATTRIBUTED max-drift spread 256K->4M = "
                f"{1000 * abs(big - small):.3f} ms "
                "(size-independence target <= 0.5 ms)"
            )


# ---------------------------------------------------------------------------
# measurement 2 — sync throughput
# ---------------------------------------------------------------------------


def bench_throughput(requests: int) -> None:
    print("\n== 2. sync throughput (300 KB JSON responses) ==")
    body = _sse_body(300 * 1024)
    server = _H1SseServer(body)
    try:
        with httpx.Client(verify=_verify_ctx(), timeout=60.0) as client:
            started = time.perf_counter()
            for _ in range(requests):
                client.post(f"{server.url}/v1/chat/completions", json={"model": "gpt-4o-mini"})
            baseline_loop = time.perf_counter() - started

        transport = RecordingTransport()
        wardex.init(
            intercept=True,
            transport=transport,
            batching=BatchingConfig(flush_interval=0.05),
        )
        try:
            with httpx.Client(verify=_verify_ctx(), timeout=60.0) as client:
                started = time.perf_counter()
                for _ in range(requests):
                    client.post(f"{server.url}/v1/chat/completions", json={"model": "gpt-4o-mini"})
                on_loop = time.perf_counter() - started
                wardex.flush()
                on_total = time.perf_counter() - started
            captured = len(transport.spans)
        finally:
            wardex.close()
        print(f"   requests:            {requests}")
        print(f"   loop wall (off):     {baseline_loop:.3f} s")
        print(f"   loop wall (on):      {on_loop:.3f} s   (target: no regression vs off + parse)")
        print(f"   loop+flush (on):     {on_total:.3f} s   (target <= 1.05 x sum of work)")
        print(f"   spans captured:      {captured} (expected {requests})")
    finally:
        server.close()


# ---------------------------------------------------------------------------
# measurement 3 — GIL release
# ---------------------------------------------------------------------------


def bench_gil(parses: int) -> None:
    from wardex_sdk._protocol import parse_llm_semantics

    print("\n== 3. GIL release during parse_llm_semantics ==")
    body = _sse_body(8 * 1024 * 1024)
    parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", body, None)  # warm

    def spin(duration: float) -> float:
        """Iterations/second of ONE fixed loop body — identical for the solo
        and the competing measurement, or the ratio measures the loop."""
        count = 0
        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            count += 1
        return count / duration

    # Solo rate.
    solo = spin(1.0)

    # Same loop, while the worker parses back to back for the whole window.
    stop = threading.Event()
    parsed = [0]

    def worker() -> None:
        while not stop.is_set():
            parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", body, None)
            parsed[0] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        competing = spin(max(1.0, parses * 0.05))
    finally:
        stop.set()
        thread.join()
    ratio = competing / solo
    print(f"   parse workload:      {parsed[0]} x 8 MB SSE during the window")
    print(f"   solo rate:           {solo:,.0f} iters/s")
    print(f"   competing rate:      {competing:,.0f} iters/s")
    print(f"   retained:            {100 * ratio:.1f} %   (target >= 90%; pre-release ~50%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke-run sizes")
    args = parser.parse_args()

    rounds = 2 if args.quick else 5
    concurrency = 8 if args.quick else 32
    requests = 100 if args.quick else 1000
    parses = 5 if args.quick else 20

    print(f"python {sys.version.split()[0]}, quick={args.quick}")
    bench_loop_drift(rounds, concurrency)
    bench_throughput(requests)
    bench_gil(parses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
