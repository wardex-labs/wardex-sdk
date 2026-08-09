"""The one place the compiled extension is imported.

`import wardex_sdk` used to raise whatever the extension raised. A wheel built
for the wrong ABI, a `.so` a container build stripped out of the image, an
sdist installed on a machine with no toolchain: each of those turned an
observability dependency into a host application that does not boot. An
observability SDK may never be the reason a process fails to start, so the
import is attempted once, here, and every other module reads the outcome
instead of repeating the import and repeating the crash.

`NATIVE_OK` is a flag and deliberately not a null-object layer. The SDK already
has a total no-op path -- every public entry point returns as soon as `_hub`
holds no client -- so the only thing degraded mode has to do is stop `init()`
from building one. Everything reachable only *through* `init()` (interceptors,
adapters, the protocol parsers, the codec) is then unreachable by construction,
which is why three `if not NATIVE_OK` tests cover the whole public surface and
a shadow implementation of the core is not needed.

Callers test the flag. They never wrap a working call in `try/except
ImportError`, because a `try` around `native.Limits(**kw)` would also catch a
genuine argument bug on the HEALTHY path and report it as "the extension is
missing" -- the one diagnosis that sends the reader in exactly the wrong
direction.

The guarantee, stated so nobody has to infer its edges: with the extension
unimportable, `import wardex_sdk` and every symbol on its `__all__` work (as
no-ops), and `wardex_sdk.transport`, `.context`, `.assembly`, `.adapters`,
`.interceptors` and `.pipeline` import; `wardex_sdk.protocol`, `.semantics`,
`transport._codec` and three `adapters/` modules still raise `ImportError`,
because they reach the core at import time. `.interceptors` moved across that
line when its package `__init__` stopped importing the TLS seam eagerly: every
seam is built inside its factory now, so importing the package no longer drags
`_ssl` -- and the core underneath it -- along. That is a boundary and not a gap: each
of them is reachable only through `init()`, which returns above, so degrading
them would change nothing a host can observe while rewriting eager bindings on
the parser hot path. `tests/test_native_absent.py` asserts both halves.

`ImportError` and not `ModuleNotFoundError`: a `.so` that was never installed
raises the subclass, but a `.so` that is present and unloadable -- the corrupt
or wrong-ABI wheel, the case a user is far more likely to hit -- raises the
base class out of `dlopen`. A handler written for the subclass alone catches
the easy half and lets the hard half kill the host.
"""

from __future__ import annotations

import importlib
from typing import Any

native: Any
NATIVE_OK: bool
NATIVE_ERROR: BaseException | None

try:
    # `importlib.import_module` and not `from . import _wardex_native`. Both
    # bind `wardex_sdk._wardex_native` on the package -- the name
    # `test_package_smoke.test_native_module_loads` asserts and the one every
    # test that drives the core directly imports -- but they report a MISSING
    # extension differently. `from . import` reaches this module while the
    # package's own `__init__` is still running, so when the submodule is not
    # there CPython falls back to an attribute lookup on the half-built package
    # and raises "cannot import name '_wardex_native' from partially
    # initialized module ... (most likely due to a circular import)". That text
    # sends whoever reads it hunting an import cycle that does not exist, in the
    # one situation where the answer is simply "the wheel has no core in it".
    native = importlib.import_module("._wardex_native", __package__)
except ImportError as exc:
    native = None
    NATIVE_OK = False
    NATIVE_ERROR = exc
else:
    NATIVE_OK = True
    NATIVE_ERROR = None


def unavailable_reason() -> str:
    """One line naming why the core is absent, for a message aimed at a human.

    Kept next to the import so every degraded-mode message quotes the same
    underlying error. "wardex is disabled" on its own sends the reader to the
    config; the `dlopen` text sends them to the wheel, which is where the
    problem is.
    """
    return f"{type(NATIVE_ERROR).__name__}: {NATIVE_ERROR}"
