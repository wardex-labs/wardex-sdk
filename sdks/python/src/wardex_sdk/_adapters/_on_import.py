"""Run a callback once the application has imported a framework, without importing it.

An adapter has to import its framework to patch it, and importing every
installed framework at `init()` makes an application that uses none of them pay
for all of them. So the registry waits: this `sys.meta_path` finder sees the
application's own import of a watched package, lets the import system load it
exactly as it would have, and calls back once the outermost module of that
package has finished executing on the importing thread. Only then is the
package whole enough for an adapter to import from.

TRANSPARENT. The finder answers with the spec the other finders give, and the
loader wrapper it puts on that spec hands the module back its real loader
before the module's code runs, so `__loader__`, `__spec__.loader`,
`importlib.resources`, `inspect.getsource` and `reload` see what they would
have. What another finder raises passes through untouched, as it would have
without this finder; a failure of wardex's own callback is counted and dropped.
One difference is visible, and only until the framework's first import:
`importlib.util.find_spec(<framework>).loader` is the wrapper, not the real
loader's type. The finder cannot tell that call from the import's own, which
goes through the same lookup. Every attribute the wrapper does not define is
read from the real loader, so only a type check sees the difference.

PER THREAD, PER PACKAGE. A package's `__init__` imports its submodules, and each
of those finishes before the package does; the callback waits for the depth of
nested executions of that package's modules on this thread to return to zero. A
namespace package has no code of its own, so the first of its modules that does
is the one whose end counts.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Any

from .._assembly import guard

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from importlib.machinery import ModuleSpec
    from types import ModuleType

__all__ = ["ImportTrigger"]


class ImportTrigger:
    """Calls `fire(package, module)` after the application imports a watched
    top-level package; `module` is the one whose execution just ended."""

    def __init__(self, fire: Callable[[str, str], None]) -> None:
        self._fire = fire
        self._watched: frozenset[str] = frozenset()
        self._local = threading.local()

    def watch(self, packages: Iterable[str]) -> None:
        """Replace the watched set. Swapped whole, so `find_spec` reads it without a lock."""
        self._watched = frozenset(packages)

    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        # The whole cost on an import of anything else: one split and one set lookup.
        top = fullname.partition(".")[0]
        if top not in self._watched:
            return None
        finding = self._finding()
        if fullname in finding:
            return None
        finding.add(fullname)
        try:
            # Not caught: what another finder raises is what the import system
            # would have raised calling it, and it is the host's to see.
            spec = self._find_elsewhere(fullname, path, target)
        finally:
            finding.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _Announcing(spec.loader, top, self)
        return spec

    def _find_elsewhere(self, fullname: str, path: Any, target: Any) -> ModuleSpec | None:
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                return spec
        return None

    def _finding(self) -> set[str]:
        finding = getattr(self._local, "finding", None)
        if finding is None:
            finding = self._local.finding = set()
        return finding

    def _depths(self) -> dict[str, int]:
        depths = getattr(self._local, "depths", None)
        if depths is None:
            depths = self._local.depths = {}
        return depths

    def entered(self, top: str) -> None:
        depths = self._depths()
        depths[top] = depths.get(top, 0) + 1

    def left(self, top: str, module: str, *, completed: bool) -> None:
        depths = self._depths()
        depth = depths.get(top, 1) - 1
        if depth > 0:
            depths[top] = depth
            return
        depths.pop(top, None)
        if completed:
            # Never into the host's import: a failure is counted and dropped.
            with guard("adapters.on_import"):
                self._fire(top, module)


class _Announcing:
    """The real loader, plus a call to the trigger once the module has executed."""

    __slots__ = ("_real", "_top", "_trigger")

    def __init__(self, real: Any, top: str, trigger: ImportTrigger) -> None:
        self._real = real
        self._top = top
        self._trigger = trigger

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create = getattr(self._real, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module: ModuleType) -> None:
        real = self._real
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.loader is self:
            spec.loader = real
        if getattr(module, "__loader__", None) is self:
            module.__loader__ = real
        self._trigger.entered(self._top)
        completed = False
        try:
            real.exec_module(module)
            completed = True
        finally:
            self._trigger.left(self._top, module.__name__, completed=completed)

    def __getattr__(self, name: str) -> Any:
        if name == "_real":  # unset only on an object `copy` built without `__init__`
            raise AttributeError(name)
        return getattr(self._real, name)
