"""Shared timing probe — refcount-based idempotent install/uninstall."""

from __future__ import annotations

import socket

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
