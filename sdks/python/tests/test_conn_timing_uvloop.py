import ssl
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import CaptureMode, SpanKind

uvloop = pytest.importorskip("uvloop", reason="optional dep; the uvloop connect path needs it")


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _verify_ctx() -> ssl.SSLContext:
    cert = Path(__file__).parent / "fixtures" / "cert.pem"
    return ssl.create_default_context(cafile=str(cert))


def test_uvloop_async_capture_populates_handshake(tls_server):
    # capture_mode=ALL: this test targets connect/handshake timing, not the
    # AGENT-mode policy gate, and the server response has no LLM semantics.
    wardex.init(transport=wardex.NoOpTransport(), intercept=True, capture_mode=CaptureMode.ALL)

    async def call() -> int:
        async with httpx.AsyncClient(verify=_verify_ctx()) as client:
            r = await client.post(f"{tls_server}/v1/messages", json={"model": "z"})
            return r.status_code

    code = uvloop.run(call())
    assert code == 200

    client = _hub.get_client()
    client._settle()  # finalization runs on the worker; settle before reading
    sp = [s for s in client._spans if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.connection_reused is False
    assert sp.transport.timing.tls_handshake_ms > 0.0


# --- plaintext HTTP on uvloop: not captured, and said so ---------------------
#
# uvloop reads and writes sockets in C, below the `socket.socket` methods the
# plaintext seam patches, so an http:// call on a uvloop loop leaves no span.
# That cannot be fixed from Python; what wardex owes the user is to say so once
# and to count it where `wardex.diagnostics()` and strict mode read. Each case
# runs in a fresh interpreter because the line is said once per process.

_PLAINTEXT_ON_UVLOOP = """
import asyncio, json, ssl, sys, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import httpx
import uvloop
import wardex_sdk as wardex

BODY = json.dumps({"id": "c1", "object": "chat.completion", "model": "m", "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, *a):
        pass

mode, scheme, fixtures = sys.argv[1], sys.argv[2], sys.argv[3]
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
if scheme == "https":
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(f"{fixtures}/cert.pem", f"{fixtures}/key.pem")
    server.socket = tls.wrap_socket(server.socket, server_side=True)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"{scheme}://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

async def call():
    async with httpx.AsyncClient(verify=f"{fixtures}/cert.pem") as client:
        for _ in range(3):
            await client.post(url, json={"model": "m", "messages": []})

async def init_inside_the_loop():
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    await call()

if mode == "inside":
    uvloop.run(init_inside_the_loop())
elif mode == "policy":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    asyncio.run(call())
else:
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    (uvloop.run if mode == "uvloop" else asyncio.run)(call())
where = {
    loss.reason: dict(loss.where) for loss in wardex.diagnostics().losses
}.get("instrumentation_unavailable", {})
print(json.dumps(where))
wardex.close()
"""


def _run_plaintext(mode: str, scheme: str = "http") -> tuple[dict[str, int], list[str]]:
    import json
    import subprocess
    import sys

    fixtures = str(Path(__file__).parent / "fixtures")
    out = subprocess.run(
        [sys.executable, "-c", _PLAINTEXT_ON_UVLOOP, mode, scheme, fixtures],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    said = [line for line in out.stderr.splitlines() if "uvloop" in line]
    return json.loads(out.stdout.strip().splitlines()[-1]), said


@pytest.mark.parametrize("mode", ["uvloop", "inside", "policy"])
def test_plaintext_http_on_uvloop_is_said_once_and_counted(mode):
    where, said = _run_plaintext(mode)
    assert len(said) == 1, said
    assert "http://" in said[0] and "https://" in said[0]
    assert "--loop asyncio" in said[0]
    assert where.get("interceptors.socket.uvloop") == 1


def test_the_standard_asyncio_loop_says_nothing_about_uvloop():
    where, said = _run_plaintext("asyncio")
    assert said == []
    assert "interceptors.socket.uvloop" not in where


@pytest.mark.parametrize("mode", ["uvloop", "inside", "policy"])
def test_https_only_on_uvloop_is_said_but_counts_no_loss(mode):
    # Nothing was lost: TLS is captured on uvloop. A loss here would fail every
    # strict-mode run of an https-only uvicorn app.
    where, said = _run_plaintext(mode, "https")
    assert len(said) == 1, said
    assert "interceptors.socket.uvloop" not in where


_NOT_HTTP_ON_UVLOOP = """
import asyncio, json
import uvloop
import wardex_sdk as wardex

class Server(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport
    def data_received(self, data):
        self.transport.write(b"+PONG\\r\\n")

class Client(asyncio.Protocol):
    def __init__(self, done):
        self.done = done
    def connection_made(self, transport):
        transport.write(b"PING\\r\\n")
    def data_received(self, data):
        self.done.set_result(data)

async def main():
    loop = asyncio.get_running_loop()
    server = await loop.create_server(Server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    done = loop.create_future()
    transport, _ = await loop.create_connection(lambda: Client(done), "127.0.0.1", port)
    assert await done == b"+PONG\\r\\n"
    transport.close()
    server.close()

wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
uvloop.run(main())
where = {
    loss.reason: dict(loss.where) for loss in wardex.diagnostics().losses
}.get("instrumentation_unavailable", {})
print(json.dumps(where))
wardex.close()
"""


def test_a_plaintext_connection_that_is_not_http_counts_no_loss_on_uvloop():
    # A database or cache spoken to in plaintext loses no HTTP call.
    import json
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", _NOT_HTTP_ON_UVLOOP], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert "interceptors.socket.uvloop" not in json.loads(out.stdout.strip().splitlines()[-1])
