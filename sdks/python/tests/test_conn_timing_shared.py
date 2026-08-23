"""Shared timing probe — refcount-based idempotent install/uninstall."""

from __future__ import annotations

import socket

import pytest

from wardex_sdk._interceptors import _conn_timing as ct


def test_shared_probe_refcount_idempotent():
    orig_connect = socket.socket.connect
    ct.install_shared_timing()
    patched_once = socket.socket.connect
    assert patched_once is not orig_connect  # patched by the first install
    ct.install_shared_timing()  # a second install does not re-patch (refcount++)
    assert socket.socket.connect is patched_once
    ct.uninstall_shared_timing()  # refcount still 1 → remains patched
    assert socket.socket.connect is patched_once
    ct.uninstall_shared_timing()  # refcount 0 → restored
    assert socket.socket.connect is orig_connect


def test_shared_store_is_singleton():
    assert ct.shared_timing_store() is ct.shared_timing_store()


@pytest.mark.xfail(
    strict=True,
    reason="the shared store honours `cap` only when it BUILDS the store, and close() leaves it",
)
def test_a_second_init_reapplies_max_connections():
    """A host that re-inits with a different bound gets the FIRST one, forever.

    `shared_timing_store(cap)` reads `cap` only on the branch that constructs
    the singleton, and nothing on the public path ever tears that singleton
    down: `uninstall_shared_timing` at refcount zero calls `clear()`, which
    empties the fileno table and leaves the module global in place, and
    `reset_shared_timing()` — the one function that nulls it — is test-only,
    reached solely from `Runtime.reset()`. `wardex.close()` runs the teardown
    path, which touches neither. So a re-init keeps the old cap for the life of
    the process, and what a host sees when the cap is too small is a spurious
    `connect_timing_unavailable` that points at no knob at all.
    """
    import wardex_sdk
    from wardex_sdk import LimitsConfig

    ct.reset_shared_timing()
    try:
        wardex_sdk.init(intercept=True, limits=LimitsConfig(max_connections=64))
        try:
            assert ct.shared_timing_store()._cap == 64
        finally:
            wardex_sdk.close()

        wardex_sdk.init(intercept=True, limits=LimitsConfig(max_connections=128))
        try:
            assert ct.shared_timing_store()._cap == 128
        finally:
            wardex_sdk.close()
    finally:
        ct.reset_shared_timing()
