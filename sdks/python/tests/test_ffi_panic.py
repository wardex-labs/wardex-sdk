"""The FFI panic boundary -- `bindings/python/src/shield.rs`.

PyO3 turns a Rust panic into `pyo3_runtime.PanicException`, which inherits
from `BaseException`. Every guard wardex puts around its own work on the
host's call path catches `Exception` -- deliberately, so `KeyboardInterrupt`
and `CancelledError` pass -- and so a panic under that default walks through
all of them and surfaces inside the host's `recv()`, after the socket has
already consumed the bytes. The shield converts the panic on the Rust side
into `NativePanic`, a `RuntimeError`, which the guards catch, count and
swallow. Three claims, one test each:

  1. a panic reaches Python as `NativePanic`, never as `PanicException`;
  2. inside the guard, a panic costs the host nothing -- its own return value
     comes back, and the failure is counted under the guard's site and under
     `ffi.panic_converted`;
  3. every entry point the module exposes runs inside `shielded`, checked by
     reading the Rust sources, so the next `#[pyfunction]` cannot forget.

`_panic_for_test` exists only in a build with the `panic-injection` cargo
feature, which the workspace `pyproject.toml` turns on for `uv sync`. A wheel
without it FAILS here rather than skipping: the likeliest cause is a native
module older than the source tree, which `uv sync --reinstall-package
wardex-sdk` fixes (AGENTS.md), and a skip would hide exactly that.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from wardex_sdk import _native, _wardex_native
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import guard

pytestmark = pytest.mark.usefixtures("fresh_counters")

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BINDINGS = _ROOT / "bindings" / "python" / "src"

_REBUILD = (
    "the native module has no `_panic_for_test`: it was built without the "
    "`panic-injection` feature the workspace pyproject.toml asks for, most "
    "likely because it predates this source tree. Run "
    "`uv sync --reinstall-package wardex-sdk`."
)


def test_injection_hook_is_present_in_the_development_build() -> None:
    assert hasattr(_wardex_native, "_panic_for_test"), _REBUILD


def test_panic_arrives_as_a_runtime_error_and_never_as_pyo3s_base_exception() -> None:
    assert issubclass(_wardex_native.NativePanic, RuntimeError)
    assert _native.NATIVE_PANIC is _wardex_native.NativePanic
    with pytest.raises(_wardex_native.NativePanic) as caught:
        _wardex_native._panic_for_test()
    # The panic's own message survives the conversion, so a converted panic
    # in the field is diagnosable from the exception text alone.
    assert "injected by _panic_for_test" in str(caught.value)
    # The property everything else rests on: `except Exception` catches it.
    assert isinstance(caught.value, Exception)
    assert "PanicException" not in [c.__name__ for c in type(caught.value).__mro__]


def test_inside_the_guard_the_host_keeps_its_result_and_the_panic_is_counted() -> None:
    # The shape of every byte-seam wrapper (`_interceptors/_ssl.py`): the
    # host's real call first, wardex's work under the guard, the host's own
    # result returned whatever wardex did.
    def recv_like(real):  # noqa: ANN001, ANN202
        ret = real()
        with guard("test.ffi_panic"):
            _wardex_native._panic_for_test()
        return ret

    assert recv_like(lambda: b"the host's bytes") == b"the host's bytes"
    assert counters.get("test.ffi_panic") == 1
    assert counters.get("ffi.panic_converted") == 1


def test_an_ordinary_failure_is_not_counted_as_a_panic() -> None:
    with guard("test.ordinary"):
        raise ValueError("not a panic")
    assert counters.get("test.ordinary") == 1
    assert counters.get("ffi.panic_converted") == 0


# --- 3. every entry point runs inside `shielded` ------------------------------

_LINE_COMMENT = re.compile(r"//[^\n]*")


def _close(src: str, k: int, open_c: str, close_c: str) -> int:
    """Index just past the bracket that closes the one at `k`."""
    depth, m = 1, k + 1
    while depth:
        c = src[m]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
        m += 1
    return m


def _fn_body(src: str, fn_at: int) -> tuple[str, str]:
    """`(name, body)` of the `fn` whose keyword starts at `fn_at`."""
    name = re.match(r"fn\s+(\w+)\s*\(", src[fn_at:])
    assert name is not None, src[fn_at : fn_at + 40]
    params_end = _close(src, fn_at + name.end() - 1, "(", ")")
    body_open = src.index("{", params_end)
    return name.group(1), src[body_open + 1 : _close(src, body_open, "{", "}") - 1]


def _entry_points(src: str) -> list[tuple[str, str, bool]]:
    """Every `(name, body, is_getter)` PyO3 exposes from one source file."""
    src = _LINE_COMMENT.sub("", src)
    out: list[tuple[str, str, bool]] = []
    for m in re.finditer(r"#\[pyfunction\]", src):
        fn_at = src.index("fn ", m.end())
        name, body = _fn_body(src, fn_at)
        out.append((name, body, False))
    for m in re.finditer(r"#\[pymethods\]", src):
        impl_open = src.index("{", m.end())
        impl_body = src[impl_open + 1 : _close(src, impl_open, "{", "}") - 1]
        cursor = 0
        for fm in re.finditer(r"(?m)^\s*(?:pub(?:\(crate\))?\s+)?fn\s+\w+\s*\(", impl_body):
            if fm.start() < cursor:
                continue  # a nested `fn` inside a body already consumed
            attrs = impl_body[cursor : fm.start()]
            fn_at = fm.start() + len(fm.group(0)) - len(fm.group(0).lstrip())
            name, body = _fn_body(impl_body, impl_body.index("fn", fn_at))
            is_getter = "#[getter]" in attrs or "#[setter]" in attrs
            out.append((name, body, is_getter))
            cursor = fm.start() + len(fm.group(0)) + len(body) + 1
    return out


def test_every_non_getter_entry_point_runs_inside_shielded() -> None:
    """HARD RULE: a `#[pyfunction]`, a `#[new]` or a `#[pymethods]` method
    that is not a `#[getter]` wraps its whole body in `shielded(...)`.

    Getters are the one exemption because nothing from the core runs in
    them: they hand back a field the constructor already owns. A getter that
    starts computing is a getter that must be shielded, and the exemption is
    by attribute rather than by name so that is a one-line change here.
    """
    unshielded: list[str] = []
    seen = getters = 0
    for path in sorted(_BINDINGS.glob("*.rs")):
        for name, body, is_getter in _entry_points(path.read_text()):
            if is_getter:
                getters += 1
                continue
            seen += 1
            if "shielded(" not in body:
                unshielded.append(f"{path.relative_to(_ROOT)}::{name}")
    # Floors, not exact counts: 29 shielded entries and 140-odd getters on
    # 2026-09-14. Both exist so a parser that quietly stops seeing one kind
    # (every `fn` read as a getter, say) fails here instead of passing vacuously.
    assert seen >= 25, f"only {seen} entry points found -- the source parser lost its grip"
    assert getters >= 100, (
        f"only {getters} getters found -- the getter exemption is not being exercised"
    )
    assert not unshielded, (
        "PyO3 entry points whose body does not run inside `shielded(...)`: "
        f"{unshielded}. A panic there reaches the host as a BaseException "
        "(see bindings/python/src/shield.rs)."
    )
