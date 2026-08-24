"""Fork-semantics benchmarks — run by hand, never collected by pytest.

    uv run python sdks/python/tests/bench_fork_semantics.py

Three numbers, matching the design's performance gates:

  1. fork round-trip delta — median os.fork() -> child os._exit(0) -> waitpid
     round trip, before wardex.init() vs after (with tables populated). The
     delta is the child hook's cost as the host experiences it; the parent
     side is hook-free by construction (no before/after_in_parent hook).
  2. child reset one-time cost — the hook's own measured duration
     (`Runtime._fork_reinit_us`) in a child with populated tables.
  3. pid stamp cost — `replace(resource_info, process_pid=os.getpid())`,
     the per-BATCH work the live stamp adds to `_drain` (never per-span).

Deliberately not a pytest test: wall-clock medians on a shared CI box are
noise, and the regression guard for the reset cost is the bounded assertion
in test_fork_semantics (`test_child_reset_is_counted_and_bounded`).
"""

from __future__ import annotations

import json
import os
import statistics
import time


def _fork_round_trip_us(n: int) -> float:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def main() -> None:
    n = int(os.environ.get("BENCH_FORKS", "1000"))

    baseline_us = _fork_round_trip_us(n)
    print(f"fork round trip, no wardex:        median {baseline_us:8.1f} µs  (n={n})")

    import wardex_sdk as wardex
    from wardex_sdk import _hub, _runtime
    from wardex_sdk._config import BatchingConfig
    from wardex_sdk._enums import CaptureMode, SpanKind, StatusCode
    from wardex_sdk._interceptors._registry import get_registry as interceptors
    from wardex_sdk._types import InternalSpan, ResourceInfo, SpanContext, SpanId, TraceId
    from wardex_sdk.testing import RecordingTransport

    wardex.init(
        transport=RecordingTransport(),
        intercept=True,
        capture_mode=CaptureMode.ALL,
        batching=BatchingConfig(flush_interval=3600.0, flush_on_signals=False),
    )

    # Populate what a working process populates, so the reset clears real state.
    seam = interceptors()._installed["ssl"]

    class _Sock:
        def selected_alpn_protocol(self):
            return "http/1.1"

        def getpeername(self):
            return ("127.0.0.1", 443)

        def fileno(self):
            return -1

    socks = [_Sock() for _ in range(32)]
    for sock in socks:
        seam._on_request_bytes(
            sock, b"POST /v1/messages HTTP/1.1\r\nHost: h\r\nContent-Length: 0\r\n\r\n"
        )
    client = _hub.get_client()
    for i in range(256):
        client.capture_span(
            InternalSpan(
                context=SpanContext(TraceId.generate(), SpanId.generate()),
                parent_span_id=None,
                name=f"buffered-{i}",
                kind=SpanKind.CLIENT,
                start_time_ns=1,
                end_time_ns=2,
                status=StatusCode.OK,
            )
        )

    installed_us = _fork_round_trip_us(n)
    print(f"fork round trip, wardex installed: median {installed_us:8.1f} µs  (n={n})")
    print(f"delta (the child hook's cost):     {installed_us - baseline_us:8.1f} µs")

    # The hook's own measured duration, reported from a child. Two figures:
    # the FIRST run (what a real child pays — dominated by cold-page and
    # refcount effects of dropping the inherited state, not by the reset's
    # own work) and a warm re-run (the reset's own work, tens of µs).
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        first = _runtime.runtime()._fork_reinit_us
        _runtime.runtime().after_in_child()
        warm = _runtime.runtime()._fork_reinit_us
        os.write(w, json.dumps({"first_us": first, "warm_us": warm}).encode())
        os.close(w)
        os._exit(0)
    os.close(w)
    payload = json.loads(os.read(r, 4096).decode())
    os.close(r)
    os.waitpid(pid, 0)
    print(f"child reset, first run:            {payload['first_us']:8d} µs (self-measured)")
    print(f"child reset, warm re-run:          {payload['warm_us']:8d} µs (the reset's own work)")

    # The per-batch pid stamp.
    from dataclasses import replace

    resource = ResourceInfo(service_name="bench", release="1", environment="e")
    m = 1_000_000
    t0 = time.perf_counter()
    for _ in range(m):
        replace(resource, process_pid=os.getpid())
    per_call_ns = (time.perf_counter() - t0) / m * 1e9
    print(f"pid stamp (per batch, not per span): {per_call_ns:6.0f} ns")

    wardex.close()


if __name__ == "__main__":
    main()
