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

Being uncollected also puts this file outside the 3.10 floor check: that script
runs pytest, so it never imports this module either, and ruff's `target-version`
sees syntax but not API availability. A 3.11+-only call added here stays green
in every automated gate and turns up only when a person runs the driver. After
editing, re-run it once under `.venv-py310/bin/python`.

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


def _filler() -> str:
    """An attribute value large enough that a handful of spans cross the cap.

    Random hex, not a repeated byte: gzip collapses repetition, and a filler
    that compresses to nothing produces a batch that fits after all -- a check
    that passes while testing the opposite of what it claims.
    """
    return secrets.token_hex(FILLER_HEX_CHARS // 2)


class _Wire:
    """What the proxy saw, and which export call produced it.

    Both lists are process-global and outlive a scenario, so every assertion
    reads them through a mark taken at the top of the scenario. Without that a
    check can be satisfied by somebody else's export -- true today only because
    the first scenario happens to run first, and silently false the moment one
    is added or reordered.
    """

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

    def export_count(self) -> int:
        with self.lock:
            return len(self.exports)

    def exports_since(self, mark: int) -> list[dict]:
        with self.lock:
            return self.exports[mark:]


WIRE = _Wire()


def _carried(body: bytes, content_encoding: str | None) -> tuple[int, list[dict]]:
    """What one request decompresses to, and which spans a receiver finds there.

    Decompressed with the standard library rather than with the core's own
    `gunzip`: a decompressor written by the same code that compressed agrees
    with itself about frames no other reader accepts, and what has to hold here
    is that somebody else can read what went out.

    The decompressed length comes back alongside the spans because the request
    cap is defined on both numbers a receiver checks -- the body on the wire and
    the message it expands to -- and the second is the binding one for base64
    payload attributes. Asserting only the compressed side would pass with a lot
    of slack exactly where the real bound is tight.
    """
    from wardex_sdk import _wardex_native

    if content_encoding == "gzip":
        body = gzip.decompress(body)
    decoded = _wardex_native.codec.decode_otlp_traces(body)
    return len(body), [
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
            encoding = self.headers.get("Content-Encoding")
            record = {
                "content_encoding": encoding,
                "content_type": self.headers.get("Content-Type"),
                "bytes": len(body),
                "receiver_status": status,
                "decompressed_bytes": None,
                "decode_error": None,
                "spans": [],
            }
            # Decoded outside WIRE.lock, and inside a try. Outside the lock
            # because a native decode is the slowest thing here and holding it
            # serializes concurrent handlers for no reason. Inside a try because
            # a raise would kill this thread before it answers, and the SDK
            # would then sit in urlopen until the 60s transport timeout and
            # report an export failure -- pointing whoever reads it at the
            # receiver rather than at this driver. Recorded before the response
            # so that `_Observed.export` never returns ahead of the record.
            try:
                record["decompressed_bytes"], record["spans"] = _carried(body, encoding)
            except Exception as exc:  # noqa: BLE001 — a decode fault is evidence, not a crash
                record["decode_error"] = repr(exc)
            with WIRE.lock:
                WIRE.requests.append(record)
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
    export_mark = WIRE.export_count()
    run = secrets.token_hex(4)
    with wardex_sdk.trace(f"e2e-root-{run}") as root:
        trace_id = root.context.trace_id.hex()
        root_span_id = root.context.span_id.hex()
        for index in range(CHILDREN):
            with wardex_sdk.span(f"e2e-child-{index}-{run}") as child:
                child.set_attribute("wardex.e2e.filler", _filler())
    wardex_sdk.flush(60.0)

    posts = WIRE.since(mark)
    exports = WIRE.exports_since(export_mark)
    split = [e for e in exports if e["requests"] > 1]
    report.check(
        bool(split),
        "one export() became several POSTs",
        ", ".join(f"{e['spans']} spans -> {e['requests']} requests" for e in exports),
    )
    report.check(
        all(p["content_encoding"] == "gzip" for p in posts),
        "every POST left gzipped",
        {p["content_encoding"] for p in posts},
    )
    report.check(
        all(p["decode_error"] is None for p in posts),
        "every POST decoded as OTLP outside the SDK",
        [p["decode_error"] for p in posts if p["decode_error"]],
    )
    report.check(
        all(p["bytes"] <= REQUEST_CAP for p in posts),
        f"every POST body stayed under the {REQUEST_CAP}-byte cap",
        [p["bytes"] for p in posts],
    )
    report.check(
        all(
            p["decompressed_bytes"] is not None and p["decompressed_bytes"] <= REQUEST_CAP
            for p in posts
        ),
        "and so did what each one decompressed to, the tighter half of the cap",
        [p["decompressed_bytes"] for p in posts],
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
        f"root in request {root_post + 1 if root_post is not None else '?'} "
        f"of {len(posts)}, "
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

    The stderr assertion counts ONE line because `report_once` is bounded to one
    line per key per process. That makes the count a statement about this
    scenario only if no earlier scenario has already spent the key, which is
    fragile in a file whose caps and filler sizes are meant to be tuned -- raise
    the filler and the first scenario starts dropping too. So the ledger is
    reset here rather than assumed empty.
    """
    import wardex_sdk
    from wardex_sdk._assembly._diag import reset_reports_for_test

    print("\nA span too large even alone is dropped without taking the batch")
    reset_reports_for_test()
    mark = WIRE.count()
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

    # Asked of the wire, not of the receiver. `_spans_at` returns the instant
    # the spans that are SUPPOSED to survive are indexed, so "the receiver does
    # not have the oversized one yet" is a statement about poll timing: had the
    # SDK exported it after all, this would print PASS for a real loss of the
    # drop guarantee. The proxy recorded every span that left the process, and
    # that is the claim -- the SDK never put it on the wire.
    on_the_wire = {span["name"] for post in WIRE.since(mark) for span in post["spans"]}
    report.check(
        oversized not in on_the_wire,
        "the oversized span never left the process",
        f"{len(on_the_wire)} span(s) on the wire",
    )

    expected = survivors | {f"e2e-drop-root-{run}"}
    stored = _spans_at(phoenix, trace_id, len(expected))
    names = {s["name"] for s in stored}
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
    """Keep the run watchable while still asserting on what was printed.

    Installed as the process-wide `sys.stderr`, deliberately: the drop is
    reported from the batch-worker thread as often as from this one, and a
    thread-local redirect would miss it.

    Everything but `write`/`flush` is delegated to the real stream. wardex's own
    stderr paths are all `print(..., file=...)` and would be satisfied by those
    two alone, but a partial file object standing in for `sys.stderr` for the
    whole interpreter is a trap for anything else on the export path that probes
    a stream the way real code does -- `fileno`, `isatty`, `encoding` -- and the
    AttributeError would surface on a background thread inside a drain.
    """

    def __init__(self, real, *streams) -> None:
        self._real = real
        self._streams = (real, *streams)
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        # Locked because the reporting thread is not always this one, and a
        # torn write would make the captured text unassertable.
        with self._lock:
            for stream in self._streams:
                stream.write(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            for stream in self._streams:
                stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._real, name)


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
    from wardex_sdk import BatchingConfig, LimitsConfig, OtlpHttpTransport

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
        # This check drives MANUAL spans at a live receiver; the byte seams
        # (now on by default) would only add unrelated traffic to the batches
        # whose composition is under test.
        intercept=False,
        limits=LimitsConfig(max_otlp_request_bytes=REQUEST_CAP),
        # The default 5s tick would be a second exporter running alongside this
        # driver, and the hazard is batch COMPOSITION rather than lock safety: a
        # tick landing between two `span()` blocks drains the batch in pieces,
        # each small enough to fit ONE request, and "one export became several
        # POSTs" then fails for a scheduler coincidence that reads exactly like
        # an SDK regression. Pushed past the whole run so the explicit flush is
        # the only export there is.
        batching=BatchingConfig(flush_interval=3600.0),
    )
    try:
        _split_export_becomes_one_trace(phoenix, report)
        _one_oversized_span_costs_one_span(phoenix, report)
    finally:
        try:
            wardex_sdk.close()
        finally:
            # `shutdown` stops the accept loop but leaves the socket listening,
            # so a late export -- the atexit hook re-entering a drain, or a
            # close() that raised partway -- would connect and then wait out the
            # full 60s transport timeout instead of failing at once.
            server.shutdown()
            server.server_close()

    print(f"\n{report.failures} failure(s)")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
