"""Manual E2E: a SPLIT OTLP export must reassemble into ONE trace at a real backend.

The in-process tests in `test_otlp_http.py` prove that an export over
`max_otlp_request_bytes` leaves as several POSTs and that every span is in one
of them. They cannot prove the property a user actually cares about, because it
is not the SDK's to decide: a receiver handed N requests must key them by trace
id and show ONE trace, not N fragments. Nothing in the SDK enforces that, so the
only way to know is to ask a real receiver.

Opt-in and never part of the suite. The filename does not match pytest's
`python_files`, so `pytest sdks/python/tests` does not collect it, and it
refuses to run without `WARDEX_E2E_PHOENIX` — CI must not depend on Docker.

    docker run -d --name wardex-e2e-phoenix -p 6006:6006 arizephoenix/phoenix:latest
    WARDEX_E2E_PHOENIX=http://127.0.0.1:6006 \
        uv run python sdks/python/tests/e2e_split_export_phoenix.py

Phoenix rather than a collector with a debug exporter because a collector only
proves the requests were *parsed*. Phoenix stores spans and serves them back
grouped by trace over `/v1/projects/default/spans`, which is the assertion this
file exists to make. That read-back path is the one Phoenix-specific thing here;
the export path is plain OTLP/HTTP.

Every POST is routed through a recording reverse proxy rather than straight at
Phoenix. Counting `urlopen` calls would only re-check what the in-process tests
already check, on the same side of the seam; the proxy sees the bytes as they
go out — how many requests, how large, and whether `Content-Encoding: gzip`
survived to the receiver that then answered 200.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Low enough that a handful of ordinary spans cross it, high enough that a
# single span still fits comfortably -- the split has to come from the BATCH
# being large, which is the case a user meets, and not from one span being
# pathological, which is the next scenario down.
REQUEST_CAP = 64 * 1024
CHILDREN = 8
FILLER_HEX_CHARS = 24_000

# Random hex, not a repeated byte: gzip collapses repetition, and a filler that
# compresses to nothing produces a batch that fits after all -- a check that
# passes while testing the opposite of what it claims.
_filler = lambda: secrets.token_hex(FILLER_HEX_CHARS // 2)  # noqa: E731


class _Wire:
    """What the proxy saw, and which export call produced it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.exports: list[dict] = []

    def count(self) -> int:
        with self.lock:
            return len(self.requests)

    def since(self, mark: int) -> list[dict]:
        with self.lock:
            return self.requests[mark:]


WIRE = _Wire()


def _carried(body: bytes, content_encoding: str | None) -> list[dict]:
    """Which spans are in one request, read the way a receiver reads it.

    Decompressed with the standard library rather than with the core's own
    `gunzip`: a decompressor written by the same code that compressed agrees
    with itself about frames no other reader accepts, and what has to hold here
    is that somebody else can read what went out.
    """
    from wardex_sdk import _wardex_native

    if content_encoding == "gzip":
        body = gzip.decompress(body)
    decoded = _wardex_native.codec.decode_otlp_traces(body)
    return [
        {
            "name": span["name"],
            "span_id": span["span_id"],
            "parent_span_id": span["parent_span_id"] or None,
        }
        for rs in decoded["resource_spans"]
        for ss in rs["scope_spans"]
        for span in ss["spans"]
    ]


def _proxy(upstream: str) -> ThreadingHTTPServer:
    """Record each POST, forward it verbatim, answer with the receiver's own status.

    Verbatim matters: re-encoding the body here would make the gzip assertion
    a statement about this proxy instead of about what the transport sent.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's spelling
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            forwarded = {
                name: value
                for name, value in self.headers.items()
                if name.lower() in ("content-type", "content-encoding")
            }
            request = urllib.request.Request(
                upstream + self.path, data=body, headers=forwarded, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    status, payload = response.status, response.read()
            except urllib.error.HTTPError as exc:
                status, payload = exc.code, exc.read()
            except Exception as exc:  # noqa: BLE001 — a proxy must answer, not raise
                status, payload = 599, str(exc).encode()
            with WIRE.lock:
                WIRE.requests.append(
                    {
                        "content_encoding": self.headers.get("Content-Encoding"),
                        "content_type": self.headers.get("Content-Type"),
                        "bytes": len(body),
                        "receiver_status": status,
                        "spans": _carried(body, self.headers.get("Content-Encoding")),
                    }
                )
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _spans_at(phoenix: str, trace_id: str, expected: int, timeout: float = 30.0) -> list[dict]:
    """Every stored span of `trace_id`, once the receiver has caught up.

    Polls rather than sleeping a fixed interval: ingestion is asynchronous on
    the receiver's side, so a fixed wait is either flaky or slow, and the exit
    condition -- `expected` spans -- is known.
    """
    deadline = time.monotonic() + timeout
    found: list[dict] = []
    while time.monotonic() < deadline:
        found = [s for s in _all_spans(phoenix) if s["context"]["trace_id"] == trace_id]
        if len(found) >= expected:
            return found
        time.sleep(1.0)
    return found


def _all_spans(phoenix: str) -> list[dict]:
    out: list[dict] = []
    cursor = None
    while True:
        url = f"{phoenix}/v1/projects/default/spans?limit=500"
        if cursor:
            url += f"&cursor={cursor}"
        with urllib.request.urlopen(url, timeout=20) as response:
            page = json.loads(response.read())
        out.extend(page["data"])
        cursor = page.get("next_cursor")
        if not cursor:
            return out


class _Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, claim: str, evidence: object = "") -> None:
        if not ok:
            self.failures += 1
        mark = "PASS" if ok else "FAIL"
        suffix = f"  [{evidence}]" if evidence != "" else ""
        print(f"  {mark}  {claim}{suffix}")


def _split_export_becomes_one_trace(phoenix: str, report: _Report) -> None:
    import wardex_sdk

    print("\nA batch over the request cap arrives as one trace")
    mark = WIRE.count()
    run = secrets.token_hex(4)
    with wardex_sdk.trace(f"e2e-root-{run}") as root:
        trace_id = root.context.trace_id.hex()
        root_span_id = root.context.span_id.hex()
        for index in range(CHILDREN):
            with wardex_sdk.span(f"e2e-child-{index}-{run}") as child:
                child.set_attribute("wardex.e2e.filler", _filler())
    wardex_sdk.flush(60.0)

    posts = WIRE.since(mark)
    split = [e for e in WIRE.exports if e["requests"] > 1]
    report.check(
        bool(split),
        "one export() became several POSTs",
        ", ".join(f"{e['spans']} spans -> {e['requests']} requests" for e in WIRE.exports),
    )
    report.check(
        all(p["content_encoding"] == "gzip" for p in posts),
        "every POST left gzipped",
        {p["content_encoding"] for p in posts},
    )
    report.check(
        all(p["bytes"] <= REQUEST_CAP for p in posts),
        f"every POST body stayed under the {REQUEST_CAP}-byte cap",
        [p["bytes"] for p in posts],
    )
    report.check(
        bool(posts) and all(p["receiver_status"] == 200 for p in posts),
        "the receiver accepted every POST with the gzip encoding as sent",
        [p["receiver_status"] for p in posts],
    )
    # The whole reason a split can lose a trace is that a parent edge now points
    # across a request boundary. If the root happened to travel with all of its
    # children, the reassembly below would hold for a batch that was never
    # really split, and the check would be worth nothing -- so make the
    # arrangement itself an assertion rather than an assumption. Spans leave in
    # completion order, so the root is genuinely last: every child arrives
    # naming a parent the receiver has not seen yet.
    root_post = next(
        (i for i, p in enumerate(posts) if any(s["span_id"] == root_span_id for s in p["spans"])),
        None,
    )
    orphaned_on_arrival = sum(
        1
        for i, post in enumerate(posts)
        for span in post["spans"]
        if span["parent_span_id"] == root_span_id and (root_post is None or i < root_post)
    )
    report.check(
        root_post is not None and orphaned_on_arrival > 0,
        "children arrived in earlier requests than the parent they name",
        f"root in request {root_post} of {len(posts)}, "
        f"{orphaned_on_arrival} child(ren) arrived before it",
    )

    expected = {f"e2e-root-{run}"} | {f"e2e-child-{i}-{run}" for i in range(CHILDREN)}
    stored = _spans_at(phoenix, trace_id, len(expected))
    names = {s["name"] for s in stored}
    report.check(
        names == expected,
        "the receiver stored every span of the split batch",
        f"{len(stored)}/{len(expected)}, missing {sorted(expected - names)}",
    )
    report.check(
        {s["context"]["trace_id"] for s in stored} == {trace_id},
        "all of them under ONE trace id",
        trace_id,
    )
    roots = [s for s in stored if s["parent_id"] is None]
    children = [s for s in stored if s["parent_id"] is not None]
    report.check(
        len(roots) == 1 and roots[0]["context"]["span_id"] == root_span_id,
        "exactly one root, and it is the span the SDK made the root",
        [s["name"] for s in roots],
    )
    report.check(
        len(children) == CHILDREN and {s["parent_id"] for s in children} == {root_span_id},
        "every child's parent edge survived the split",
        sorted({s["parent_id"] for s in children}),
    )


def _one_oversized_span_costs_one_span(phoenix: str, report: _Report) -> None:
    """The failure edge. A span that cannot fit a request even alone is dropped,
    and the rest of its batch must still arrive.

    It is dropped rather than marked because the marker mechanism cannot reach
    it: a `wardex.limitations` marker rides on the span, and this span never
    reaches the wire to carry one. The bounded stderr line is the only channel
    it has, so it is part of the documented behavior and is asserted here.
    """
    import wardex_sdk

    print("\nA span too large even alone is dropped without taking the batch")
    run = secrets.token_hex(4)
    survivors = {f"e2e-small-{i}-{run}" for i in range(3)}
    oversized = f"e2e-oversized-{run}"

    captured = io.StringIO()
    with contextlib.redirect_stderr(_Tee(sys.stderr, captured)):
        with wardex_sdk.trace(f"e2e-drop-root-{run}") as root:
            trace_id = root.context.trace_id.hex()
            for index in range(3):
                with wardex_sdk.span(f"e2e-small-{index}-{run}"):
                    pass
            with wardex_sdk.span(oversized) as big:
                # Deliberately an attribute and not captured payload: payload is
                # what the encoder strips as its last resort before giving up,
                # so a payload-only span would exercise the recovery rather than
                # the drop.
                big.set_attribute("wardex.e2e.filler", secrets.token_hex(REQUEST_CAP * 4))
        wardex_sdk.flush(60.0)
    diagnostics = captured.getvalue()

    expected = survivors | {f"e2e-drop-root-{run}"}
    stored = _spans_at(phoenix, trace_id, len(expected))
    names = {s["name"] for s in stored}
    report.check(
        oversized not in names,
        "the oversized span did not reach the receiver",
    )
    report.check(
        names == expected,
        "every other span of the same batch did",
        f"{sorted(names)}",
    )
    said_so = [
        line
        for line in diagnostics.splitlines()
        if "exceeded max_otlp_request_bytes" in line and "not exported" in line
    ]
    report.check(
        len(said_so) == 1,
        "the drop was reported once on stderr, off debug",
        said_so or diagnostics.splitlines()[-3:],
    )


class _Tee:
    """Keep the run watchable while still asserting on what was printed."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> int:
    phoenix = os.environ.get("WARDEX_E2E_PHOENIX")
    if not phoenix:
        print(
            "WARDEX_E2E_PHOENIX is not set — skipping. This check needs a live OTLP "
            "receiver (see the module docstring for the docker one-liner) and is "
            "opt-in so that no suite depends on Docker."
        )
        return 0
    phoenix = phoenix.rstrip("/")
    try:
        with urllib.request.urlopen(f"{phoenix}/v1/projects", timeout=10):
            pass
    except Exception as exc:  # noqa: BLE001 — a missing receiver is a message, not a stack
        print(f"{phoenix} is not answering ({exc}) — start the receiver first")
        return 2

    import wardex_sdk
    from wardex_sdk import CaptureLimits, OtlpHttpTransport

    server = _proxy(phoenix)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/traces"
    print(f"receiver {phoenix}, recording proxy {endpoint}, cap {REQUEST_CAP} bytes")

    class _Observed(OtlpHttpTransport):
        """Attributes POSTs to the export that produced them.

        Without this the proxy can say "five requests arrived" but not "ONE
        envelope became five", and the second is the claim.
        """

        def export(self, envelope, *, timeout=None):
            before = WIRE.count()
            result = super().export(envelope, timeout=timeout)
            with WIRE.lock:
                WIRE.exports.append(
                    {"spans": len(envelope.spans), "requests": len(WIRE.requests) - before}
                )
            return result

    report = _Report()
    wardex_sdk.init(
        transport=_Observed(endpoint=endpoint, timeout=60.0),
        limits=CaptureLimits(max_otlp_request_bytes=REQUEST_CAP),
    )
    try:
        _split_export_becomes_one_trace(phoenix, report)
        _one_oversized_span_costs_one_span(phoenix, report)
    finally:
        wardex_sdk.close()
        server.shutdown()

    print(f"\n{report.failures} failure(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
