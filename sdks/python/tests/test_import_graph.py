"""Structural conformance — design §3.1 (dependency direction) and §10.4 (C-S1..C-S5).

These tests read the source with `ast`; they never import the modules they
judge. They exist because of the single most instructive failure in the prior
art: an emitter helper was *extracted* into one place, and then a second
constructor and eight streaming branches were left free to bypass it. An
extraction that leaves the bypass reachable is not an extraction. So the rules
below are the extraction — the docstrings in `_assembly/` are only its
description.

Two kinds of assertion live here, and the difference matters when one fails:

  HARD RULE — true today, must stay true forever. If you broke one, the fix is
  in your change, not here.

  BUDGET — a ratchet over code that predates the `_assembly/` package. Each file
  gets the count it has today; the test fails if a count goes UP or if a file
  not on the list acquires its first occurrence. These numbers may only be
  lowered. Each budget is driven to zero as the code it covers moves onto
  `_assembly/`, at which point the budget dict is deleted and the rule becomes
  hard. Adding a line to
  a budget to make your change pass is the one edit this file is not for.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_PKG = "wardex_sdk"


# --------------------------------------------------------------------------
# source access
# --------------------------------------------------------------------------


def _modules() -> dict[str, ast.Module]:
    """Every module in the package, keyed by path relative to `wardex_sdk/`."""
    out: dict[str, ast.Module] = {}
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        out[rel] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return out


def _known_modules() -> set[str]:
    """Dotted names of every module and package inside `wardex_sdk`.

    Used to tell `from .._types import TraceId` (an edge to `wardex_sdk._types`,
    plus a symbol) apart from `from .. import _hub` (an edge to
    `wardex_sdk._hub`). Without it the symbol would be mistaken for a module.
    """
    names = {_PKG}
    for path in _SRC.rglob("*.py"):
        parts = list(path.relative_to(_SRC).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        names.add(".".join([_PKG, *parts]))
    return names


def _known_packages() -> set[str]:
    """Dotted names of every package (directory with `__init__.py`) in wardex_sdk.

    A symbol imported from a package may be a re-export that hides the module
    that defines it; a symbol imported from a leaf module cannot.
    """
    return {
        ".".join([_PKG, *path.relative_to(_SRC).parts[:-1]]) for path in _SRC.rglob("__init__.py")
    }


def _imported_modules(rel: str, tree: ast.Module) -> set[str]:
    """Absolute dotted names of every wardex module `rel` imports.

    Relative imports are resolved against `rel`'s own package, and
    `from .. import _hub` is resolved to `wardex_sdk._hub` rather than left as
    a bare name — the rule is about the module graph, not about spelling.

    A symbol pulled out of a PACKAGE — `from .. import trace`, where `trace` is
    re-exported by `wardex_sdk/__init__.py` rather than being a module — yields
    the pseudo-target `wardex_sdk:trace`. Such an import produces no module edge
    at all (the defining module is `_tracing.py`, which is never named), so
    without this a layering rule that forbids `wardex_sdk._tracing` is one
    re-export away from meaning nothing. The colon keeps it un-confusable with a
    real dotted name, so an allowlist has to name it deliberately. Symbols taken
    from a leaf MODULE need no such treatment: `from .._types import TraceId`
    already carries the auditable edge `wardex_sdk._types`.

    Dynamic imports (`importlib.import_module("wardex_sdk.x")`, `__import__`)
    are resolved too when the argument is a string literal, and reported as
    `<dynamic>` when it is not: `_adapters/__init__.py` already imports
    `importlib.util`, and design §3.2 gives `_assembly/_patchset.py` the job of
    reaching modules lazily, so this idiom is one edit away from being a hole in
    every rule below.
    """
    known = _known_modules()
    packages = _known_packages()
    pkg = [_PKG, *rel.split("/")[:-1]]  # package containing this module
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                target = [*base, *(node.module.split(".") if node.module else [])]
            elif node.module and (node.module == _PKG or node.module.startswith(_PKG + ".")):
                target = node.module.split(".")
            else:
                continue  # stdlib or third party
            dotted = ".".join(target)
            if dotted in known:
                found.add(dotted)
            for alias in node.names:
                # `from .. import _hub` — the imported name may itself be a module.
                candidate = f"{dotted}.{alias.name}"
                if candidate in known:
                    found.add(candidate)
                elif alias.name != "*" and dotted in packages:
                    found.add(f"{dotted}:{alias.name}")
        elif isinstance(node, ast.Call) and _is_dynamic_import(node):
            found.update(_dynamic_import_target(node))
    return found


def _is_dynamic_import(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "__import__":
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id in ("importlib", "il")
    )


def _dynamic_import_target(node: ast.Call) -> set[str]:
    if not node.args:
        return set()
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        name = first.value
        if name == _PKG or name.startswith(_PKG + "."):
            return {name}
        return set()
    return {"<dynamic>"}  # a non-literal target cannot be audited at all


def _imported_names(tree: ast.Module) -> set[str]:
    """Names this module pulls in by name, including under TYPE_CHECKING.

    A type-only import still puts the symbol in the adapter author's
    vocabulary, and C-S1 is about making the bypass unnameable (I5).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            names.update(a.name.rsplit(".", 1)[-1] for a in node.names)
    return names


def _local_names(tree: ast.Module, symbol: str) -> set[str]:
    """Every local name bound to `symbol` by an import, including aliases.

    `from .._types import TraceId as _T` must not launder a rule. The scan is
    per-module because the alias is.
    """
    names = {symbol}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names if a.name == symbol)
    return names


def _module_aliases(rel: str, tree: ast.Module) -> set[str]:
    """Every local name bound to a wardex MODULE object, including aliases.

    A rule that only understands `TraceId.generate()` is defeated by
    `_types.TraceId.generate()`, and `from .. import _hub, _wardex_native` is the
    house form in this codebase (`_adapters/_assembler.py:17`,
    `_interceptors/_mcp_stdio.py:20`, `_interceptors/_trackers.py:14`) — so that
    spelling is one token away from code already in the tree. The predicates
    below therefore accept a module-qualified receiver as well as a bare name.

    Resolution is the real thing, not a naming heuristic: a name is collected
    only when the module it would bind actually exists in the package. `_PKG` is
    always included so `import wardex_sdk` + `wardex_sdk._types.TraceId` is
    covered without needing the intermediate binding.
    """
    known = _known_modules()
    pkg = [_PKG, *rel.split("/")[:-1]]
    names = {_PKG}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    # `import wardex_sdk._types` binds `wardex_sdk`;
                    # `import wardex_sdk._types as t` binds `t`.
                    names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - node.level + 1]
                target = [*base, *(node.module.split(".") if node.module else [])]
            elif node.module and (node.module == _PKG or node.module.startswith(_PKG + ".")):
                target = node.module.split(".")
            else:
                continue
            dotted = ".".join(target)
            for alias in node.names:
                if f"{dotted}.{alias.name}" in known:
                    names.add(alias.asname or alias.name)
    return names


def _reaches_symbol(node: ast.expr, symbol: str, bound: set[str], modules: set[str]) -> bool:
    """True if `node` names `symbol`, however it is spelled.

    Accepts `TraceId` and its import aliases (`bound`), plus any module-qualified
    path ending in `.TraceId` whose leftmost receiver is a wardex module name
    (`_types.TraceId`, `wardex_sdk._types.TraceId`). The rule is about reaching
    the type, not about how the reach is written.
    """
    if isinstance(node, ast.Name):
        return node.id in bound
    if not (isinstance(node, ast.Attribute) and node.attr == symbol):
        return False
    receiver: ast.expr = node.value
    while isinstance(receiver, ast.Attribute):
        receiver = receiver.value
    return isinstance(receiver, ast.Name) and receiver.id in modules


def _calls_traceid_generate(rel: str, tree: ast.Module):
    bound = _local_names(tree, "TraceId")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
            and _reaches_symbol(node.func.value, "TraceId", bound, modules)
        )

    return predicate


def _constructs_internal_span(rel: str, tree: ast.Module):
    bound = _local_names(tree, "InternalSpan")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and _reaches_symbol(
            node.func, "InternalSpan", bound, modules
        )

    return predicate


def _passes_parent_span_id(rel: str, tree: ast.Module):
    def predicate(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and any(
            kw.arg == "parent_span_id" for kw in node.keywords
        )

    return predicate


def _constructs_ambient(rel: str, tree: ast.Module):
    def predicate(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and _reaches_symbol(
            node.func, "Ambient", _local_names(tree, "Ambient"), _module_aliases(rel, tree)
        )

    return predicate


def _resolves_observed_without_asking(rel: str, tree: ast.Module):
    """A `resolve_observed(...)` call that does NOT declare `parent_closed`."""
    bound = _local_names(tree, "resolve_observed")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _reaches_symbol(node.func, "resolve_observed", bound, modules)
            and not any(kw.arg == "parent_closed" for kw in node.keywords)
        )

    return predicate


def _resolves_observed_without_declaring_eviction(rel: str, tree: ast.Module):
    """A `resolve_observed(...)` call that does NOT declare `parent_evicted`."""
    bound = _local_names(tree, "resolve_observed")
    modules = _module_aliases(rel, tree)

    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _reaches_symbol(node.func, "resolve_observed", bound, modules)
            and not any(kw.arg == "parent_evicted" for kw in node.keywords)
        )

    return predicate


def _reads_capture_mode(rel: str, tree: ast.Module):
    """Every way a module can reach `config.capture_mode`.

    Two spellings, because the codebase already uses both: the attribute
    (`client.config.capture_mode`, which is how the seam used to read it)
    and the string (`getattr(config, "capture_mode", None)`, which is how
    `_assembly/_policy.py` reads it now, duck-typed). A rule that saw only the
    first would be one `getattr` away from meaning nothing.

    The DECLARATION is deliberately not a read: `_config.py`'s
    `capture_mode: CaptureMode = CaptureMode.AGENT` is an AnnAssign onto a
    Name, and `init(**config_kwargs)` never names it at all.
    """

    def predicate(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "capture_mode"
        return isinstance(node, ast.Constant) and node.value == "capture_mode"

    return predicate


def _defines_a_capture_predicate(rel: str, tree: ast.Module):
    """`def should_capture` / `def _should_capture`, however it is spelled."""

    def predicate(node: ast.AST) -> bool:
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith(
            "should_capture"
        )

    return predicate


def _calls_sink(rel: str, tree: ast.Module):
    def predicate(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("capture_span", "capture_snapshot")
        )

    return predicate


# Calls that make a handler self-reporting. `guard` is the sanctioned swallow;
# `bump` is its counter; the rest are the logging spellings already in the tree
# (`_interceptors/_seam.py` `parser_disable_log`, stdlib logging, debug prints).
_REPORTING_CALLS = frozenset(
    {
        "guard",
        "bump",
        "parser_disable_log",
        "print",
        "log",
        "debug",
        "info",
        "warn",
        "warning",
        "error",
        "exception",
        "critical",
        # _assembly._diag's own channel: the logger spellings above, behind the
        # `wardex_sdk` logger and its fail-safe emission.
        "diag_info",
        "diag_warning",
        "report_once",
    }
)


def _is_silent_swallow(rel: str, tree: ast.Module):
    # Needs no per-module context (no import can alias `except: pass`), but keeps
    # the predicate-factory shape the other rules use so _tally stays uniform.
    return _silent_swallow_node


def _suppress_items(node: ast.AST) -> list[ast.Call]:
    """The `contextlib.suppress(...)` calls a `with` statement enters, if any.

    Matched on the callee NAME, so both spellings the stdlib offers count
    (`contextlib.suppress(E)` and a bare `suppress(E)` after `from contextlib
    import suppress`) and nothing else does. `_suppress.py` exports a context
    manager called `suppress_capture`, which suppresses CAPTURE rather than an
    exception and is not this.
    """
    if not isinstance(node, (ast.With, ast.AsyncWith)):
        return []
    found = []
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "suppress":
            found.append(call)
    return found


def _silent_swallow_node(node: ast.AST) -> bool:
    """A construct that discards an exception and leaves NO trace it ran.

    Deliberately not "the body is literally `pass`". Two things go wrong with the
    literal reading. It misses `except Exception: return None`, which is the same
    swallow with different punctuation and is already the shape of 13 handlers in
    `_adapters/`/`_interceptors/`. Worse, it makes the C-S4 ratchet satisfiable by
    laundering: rewriting `pass` as `return None` lowers the count, turns the
    budget green, and fixes nothing — which would defeat the conversion onto
    `guard()` this budget exists to drive.

    So the question asked is the one the rule's WHY actually asks: after this
    handler runs, is there any evidence anywhere? A handler is NOT silent if it
    re-raises (the failure reaches the host) or calls something that records
    (`guard`, `counters.bump`, any logging spelling). Everything else — `pass`,
    `return`, `return b""`, `self._disabled = True` — is a swallow that is
    indistinguishable from wardex not being installed.

    `contextlib.suppress(...)` is the SECOND spelling of the same act, and the
    rule would go blind the moment someone reached for it: a `try/except` and a
    `with suppress` compile to the same discarded exception, and only one of
    them used to be visible here. It gets no escape hatch, because it CANNOT
    report. An `except` body runs only when the failure happened, so a
    `counters.bump` there is evidence about that failure; a `with suppress` body
    runs on the success path and the exception unwinds straight out of it, so
    nothing written inside can say the swallow occurred. Every
    `contextlib.suppress` is therefore silent by construction, and the one
    sanctioned occurrence in `_assembly/` is recorded by name in
    `_ASSEMBLY_SUPPRESS` rather than exempted by predicate.
    """
    if _suppress_items(node):
        return True
    if not isinstance(node, ast.ExceptHandler):
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Raise):
            return False
        if isinstance(child, ast.Call):
            func = child.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _REPORTING_CALLS:
                return False
    return True


def _tally(make_predicate, *, under: tuple[str, ...] | None = None) -> dict[str, int]:
    """{module path: occurrences}, omitting zeros. `make_predicate` is built per module."""
    counts: dict[str, int] = {}
    for rel, tree in _modules().items():
        if under is not None and not rel.startswith(under):
            continue
        predicate = make_predicate(rel, tree)
        n = sum(1 for node in ast.walk(tree) if predicate(node))
        if n:
            counts[rel] = n
    return counts


def _assert_within_budget(
    actual: dict[str, int], budget: dict[str, int], rule: str, why: str
) -> None:
    over = {rel: (n, budget.get(rel, 0)) for rel, n in actual.items() if n > budget.get(rel, 0)}
    assert not over, (
        f"{rule} regressed.\n"
        + "\n".join(f"  {rel}: {n} occurrences, budget {b}" for rel, (n, b) in sorted(over.items()))
        + f"\n\nWHY: {why}\n"
        "These budgets are a ratchet over code that predates wardex_sdk._assembly.\n"
        "They may only be LOWERED. If you need a new occurrence, you need the\n"
        "_assembly/ entry point instead — that is the whole point of the rule.\n"
        "See design §10.4."
    )


# --------------------------------------------------------------------------
# design §3.1 — dependency direction (HARD RULES)
# --------------------------------------------------------------------------

# _assembly/ sits above the leaf vocabulary and the scope layer and below
# everything that observes anything. This is a positive allowlist so that a new
# module in the package forces an explicit decision instead of silently
# widening the layer.
_ASSEMBLY_MAY_IMPORT = frozenset(
    {
        f"{_PKG}",  # exact: the package object, e.g. `from .. import _hub`
        f"{_PKG}._types",
        f"{_PKG}._enums",
        f"{_PKG}._limits",
        f"{_PKG}._hash",
        f"{_PKG}._config",
        f"{_PKG}._scope",
        f"{_PKG}._hub",
        f"{_PKG}._client",
        # A stdlib-only leaf at the package root, like `_hash`. `_diag`'s
        # diagnostic channel enters it around every logger emission so a host
        # log handler that POSTs cannot have its traffic captured by the seams.
        f"{_PKG}._suppress",
    }
)

# Whole subpackages _assembly/ may reach into. Deliberately separate from the
# exact list: `wardex_sdk` itself must NOT act as a prefix, or every module in
# the SDK would be allowed and the rule would assert nothing.
_ASSEMBLY_MAY_IMPORT_PACKAGES = (f"{_PKG}.context",)

_LAYERING_WHY = (
    "_assembly/ is the layer everything else depends ON. If it may import an\n"
    "observer (_interceptors/, _adapters/, _protocol/, _semantics/), the one-way\n"
    "arrow in design §3.1 becomes a cycle, the 'adapters cannot reach\n"
    "InternalSpan' guarantee (I5) becomes reachable through assembly, and the\n"
    "package stops being a boundary anyone can reason about."
)


def test_assembly_imports_only_leaf_and_scope_layers():
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith("_assembly/"):
            continue
        for target in sorted(_imported_modules(rel, tree)):
            if target.startswith(f"{_PKG}._assembly"):
                continue
            if target in _ASSEMBLY_MAY_IMPORT:
                continue
            if any(
                target == pkg or target.startswith(pkg + ".")
                for pkg in _ASSEMBLY_MAY_IMPORT_PACKAGES
            ):
                continue
            violations.append(f"  {rel} imports {target}")
    assert not violations, (
        "_assembly/ reached above its layer:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: "
        + _LAYERING_WHY
        + "\n\nIf the dependency is genuinely a leaf, add it to"
        " _ASSEMBLY_MAY_IMPORT in this file and say why in the PR."
    )


def test_adapters_and_interceptors_do_not_import_each_other():
    violations: list[str] = []
    for a, b in (("_adapters/", f"{_PKG}._interceptors"), ("_interceptors/", f"{_PKG}._adapters")):
        for rel, tree in _modules().items():
            if not rel.startswith(a):
                continue
            for target in sorted(_imported_modules(rel, tree)):
                if target == b or target.startswith(b + "."):
                    violations.append(f"  {rel} imports {target}")
    assert not violations, (
        "_adapters/ and _interceptors/ are siblings, not a stack:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: they are two independent observers of the same run. A direct\n"
        "edge between them is how one layer ends up owning the other's spans,\n"
        "which is exactly the double-instrumentation mess design §8 exists to\n"
        "prevent. Anything they need to share belongs in _assembly/."
    )


_BELOW_THE_OBSERVERS = ("transport/", "context/")

# Its own reason, not assembly's: _LAYERING_WHY argues from _assembly/ being the
# layer everything depends ON, and a transport/ violation printing that would
# name a package the violation has nothing to do with.
_BELOW_THE_OBSERVERS_WHY = (
    "transport/ and context/ sit BELOW the observers. A span reaches the\n"
    "transport after _interceptors/ and _adapters/ are done with it, and the\n"
    "context layer cross-cuts them rather than depending on either, so an edge\n"
    "upward turns the one-way arrow in design §3.1 into a cycle. It also files\n"
    "the shared thing under the package whose behaviour it changes, which is\n"
    "how the exporter's own self-exclusion guard — read by transport/ on every\n"
    "outbound batch — ended up living inside _interceptors/."
)


def test_the_layers_below_the_observers_do_not_import_one():
    """design §3.1's arrow, in the two places it used to point backwards.

    `transport/` and `context/` sit BELOW `_interceptors/` and `_adapters/` —
    a span reaches the transport after the observers are done with it, and the
    context layer cross-cuts them rather than depending on either. Both
    nonetheless imported `interceptors._exclusion`, for the same reason: the
    self-exclusion guard was filed under the package whose behaviour it changes
    instead of the layer both of its ENDS live in. It is `_suppress.py` at the
    package root now, and this is the rule that keeps the next shared flag from
    being filed the same way.

    Scoped to the two packages rather than to every module outside
    `_interceptors/`, because `__init__.py` and `_runtime.py` are the
    composition root: installing an interceptor is what they are FOR, and a
    rule that forbade it would be a rule about the wrong thing.
    """
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith(_BELOW_THE_OBSERVERS):
            continue
        for target in sorted(_imported_modules(rel, tree)):
            for observer in (f"{_PKG}._interceptors", f"{_PKG}._adapters"):
                if target == observer or target.startswith(observer + "."):
                    violations.append(f"  {rel} imports {target}")
    assert not violations, (
        "a layer below the observers reached up into one:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: "
        + _BELOW_THE_OBSERVERS_WHY
        + "\n"
        "Anything transport/ or context/ genuinely shares with an observer is\n"
        "not an observer concern — put it at the package root, where both ends\n"
        "of it live."
    )


# _semantics/ answers "what does this parsed body mean in gen_ai terms". It reads
# the protocol layer and writes assembly's closed vocabularies, and that is the
# whole of it. A positive allowlist for the same reason assembly has one: a new
# module in the package must force a decision rather than silently widen the
# layer.
_SEMANTICS_MAY_IMPORT = frozenset(
    {
        f"{_PKG}",
        f"{_PKG}._types",
        f"{_PKG}._enums",
    }
)
_SEMANTICS_MAY_IMPORT_PACKAGES = (f"{_PKG}._assembly", f"{_PKG}._protocol")


def test_semantics_imports_neither_sibling_observer():
    """The rule the package's own docstring asserts, made checkable.

    Before this existed, `_semantics/__init__.py` claimed it "imports NEITHER
    sibling" and nothing enforced it — the extraction was a sentence. Adding
    `from .._interceptors._trackers import _Txn` to `_semantics/_grpc.py` — the
    single edit that destroys the reason the package exists — left every test in
    this file green.

    It matters because the annotation is the visible half of the same problem:
    `build_grpc_fields` takes `txn: Any` PRECISELY because naming `_Txn` would
    be this import. Without the rule, the weaker type buys nothing and the next
    author simply adds the import back.
    """
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith("_semantics/"):
            continue
        for target in sorted(_imported_modules(rel, tree)):
            if target.startswith(f"{_PKG}._semantics"):
                continue
            if target in _SEMANTICS_MAY_IMPORT:
                continue
            # `pkg + ":"` is the pseudo-target `_imported_modules` produces for a
            # symbol re-exported by a PACKAGE (`from .._assembly import
            # Limitation`). Allowed here, and only here: that IS assembly's and
            # protocol's declared public surface, which is exactly what a layer
            # below is supposed to consume. The colon form stays opt-in per
            # package, so a re-export cannot smuggle in a module this layer may
            # not reach.
            if any(
                target == pkg or target.startswith(pkg + ".") or target.startswith(pkg + ":")
                for pkg in _SEMANTICS_MAY_IMPORT_PACKAGES
            ):
                continue
            violations.append(f"{rel} -> {target}")

    assert not violations, (
        f"_semantics/ reached outside its layer: {violations}\n\n"
        "WHY: _semantics/ exists so a SECOND byte seam — and eventually a second\n"
        "language SDK — can reuse the protocol-to-gen_ai mapping without\n"
        "dragging in the interceptor that happens to call it today. One import\n"
        "of _interceptors/ or _adapters/ makes the package a private helper of\n"
        "that caller again, and the extraction was for nothing."
    )


def test_semantics_internal_modules_stay_private():
    public = sorted(
        rel
        for rel in _modules()
        if rel.startswith("_semantics/")
        and not rel.rsplit("/", 1)[-1].startswith("_")
        and rel != "_semantics/__init__.py"
    )
    assert not public, f"_semantics/ modules must stay underscore-private: {public}"


def test_semantics_public_surface_is_declared_and_resolvable():
    """The move promoted four module-private names onto a public package.

    `from wardex_sdk._semantics import build_grpc_fields` is now an import path a
    user can pin, on a package already published to PyPI. That surface gets the
    same two rules assembly's does, rather than being de-facto conforming and
    de-jure unenforced.
    """
    import wardex_sdk._semantics as semantics

    assert semantics.__all__, "semantics must declare __all__ — it is the stable surface"
    assert list(semantics.__all__) == sorted(semantics.__all__), "keep __all__ sorted"
    missing = [name for name in semantics.__all__ if not hasattr(semantics, name)]
    assert not missing, f"semantics.__all__ names nothing importable: {missing}"


def test_assembly_internal_modules_stay_private():
    public = sorted(
        rel
        for rel in _modules()
        if rel.startswith("_assembly/")
        and not rel.rsplit("/", 1)[-1].startswith("_")
        and rel != "_assembly/__init__.py"
    )
    assert not public, (
        f"_assembly/ modules must stay underscore-private: {public}\n\n"
        "WHY: assembly.__all__ is a semver-stable boundary (design §3.1). A\n"
        "module without a leading underscore invites `from wardex_sdk._assembly\n"
        "import somemodule`, which freezes an internal layout we intend to keep\n"
        "moving as more of the SDK routes through this package."
    )


def test_assembly_public_surface_is_declared_and_resolvable():
    import wardex_sdk._assembly as assembly

    assert assembly.__all__, "assembly must declare __all__ — it is the stable surface"
    assert list(assembly.__all__) == sorted(assembly.__all__), "keep __all__ sorted"
    missing = [name for name in assembly.__all__ if not hasattr(assembly, name)]
    assert not missing, f"assembly.__all__ names nothing importable: {missing}"


# --------------------------------------------------------------------------
# C-S1 — _adapters/ cannot name the span machinery
# --------------------------------------------------------------------------

_FORBIDDEN_IN_ADAPTERS = frozenset(
    {"InternalSpan", "SpanContext", "TraceId", "SpanId", "Client", "_hub", "capture_state_snapshot"}
)

# The same rule at module granularity, and it is the half that carries the
# weight. A symbol-name check is defeated by one line — `from .. import _types`
# binds no forbidden NAME, and `_types` is where InternalSpan, SpanContext,
# TraceId and SpanId all live, one attribute access away. That spelling is the
# house form here (`_adapters/_assembler.py:17`, `_interceptors/_mcp_stdio.py:20`,
# `_interceptors/_trackers.py:14`), so it is not a contrived bypass. `_hub` was
# caught before only by the accident of being spelled like a symbol.
#
# `wardex_sdk` itself is on the list: `import wardex_sdk` hands an adapter the
# whole package object, and with it every private module by attribute.
#
# `_runtime` joined the list when the process state got one owner. `_hub` is on
# it because `_hub.get_client()` is how an adapter would reach the `Client`, and
# that function is now a one-line delegate to `runtime().client` — so the two
# modules are equivalent for this rule's purpose and listing only one of them
# would leave `from .._runtime import runtime` as an unguarded way to the same
# object.
_FORBIDDEN_MODULES_IN_ADAPTERS = frozenset(
    {
        _PKG,
        f"{_PKG}._types",
        f"{_PKG}._hub",
        f"{_PKG}._runtime",
        f"{_PKG}._client",
        f"{_PKG}._tracing",
        f"{_PKG}._scope",
    }
)

# Pre-assembly debt. Each entry is the set of forbidden names and module targets
# that file still reaches today; the sets may only SHRINK. The adapter rewrite
# empties it, when the Anthropic adapter moves onto AdapterContext + _assembly/
# and this dict is deleted along with the budget machinery.
_CS1_DEBT: dict[str, frozenset[str]] = {
    "_adapters/__init__.py": frozenset({"Client", f"{_PKG}._client"}),
    "_adapters/_base.py": frozenset({"Client", f"{_PKG}._client"}),
    # `_runtime` is declared, not tolerated by omission: the registry reaches
    # the runtime only to find its own owner (`runtime().adapters`), which is
    # the one edge the ownership change requires and the only one in `_adapters/`.
    # Written down so the NEXT adapter module that reaches `_runtime` — and with
    # it `runtime().client` — is an explicit decision instead of a silent one.
    "_adapters/_registry.py": frozenset({"Client", f"{_PKG}._client", f"{_PKG}._runtime"}),
    # The unit registry shrank this by one: `wardex_sdk._tracing` is gone, because the
    # in-process tool wrapper no longer opens a MANUAL span through the public
    # tracing API. It opens a CALL unit instead, so the tool span is a child of
    # the session by construction rather than a root that happened to be started
    # inside one — the parent edge now comes from the unit that owns the call,
    # not from whatever the ambient scope happened to hold when the manual span
    # was opened.
    # What is left is the `Client` type (an install() parameter annotation) and
    # `_types` for the typed blocks.
    "_adapters/_anthropic_agent_sdk.py": frozenset({"Client", f"{_PKG}._client", f"{_PKG}._types"}),
    # Shrank by four when the assembler stopped minting ids and reading the
    # scope: `TraceId`, `SpanId`, `_hub` and `wardex_sdk._hub` are gone, because
    # `assembly.resolve_parentage()`/`child_of()` do both.
    # Shrank by two more when it stopped constructing spans and holding a bare
    # anchor context: `InternalSpan` and `SpanContext` are gone, because
    # `assembly.SpanDraft` does both and `draft.context` IS the anchor. What is
    # left is the `wardex_sdk` package object (`_wardex_native` for the limits
    # defaults) and `_types` for the typed attribute blocks.
    "_adapters/_assembler.py": frozenset(
        {
            _PKG,
            f"{_PKG}._types",
        }
    ),
}


def test_adapters_do_not_import_span_machinery():
    violations: list[str] = []
    for rel, tree in _modules().items():
        if not rel.startswith("_adapters/"):
            continue
        used = _imported_names(tree) & _FORBIDDEN_IN_ADAPTERS
        used |= _imported_modules(rel, tree) & _FORBIDDEN_MODULES_IN_ADAPTERS
        new = used - _CS1_DEBT.get(rel, frozenset())
        if new:
            violations.append(f"  {rel} imports {sorted(new)}")
    assert not violations, (
        "C-S1: an adapter module named the span machinery:\n"
        + "\n".join(sorted(violations))
        + "\n\nWHY: this is the product claim, enforced structurally. An adapter\n"
        "that can reach TraceId/SpanContext/InternalSpan/Client can mint a\n"
        "parent edge out of a framework's run id, and the causal tree stops\n"
        "coming from in-process context propagation. Forbidding the IMPORT is\n"
        "stronger than auditing call sites: with the type absent from the\n"
        "module's namespace the bypass has no name to call (design I5, §10.4).\n"
        "Adapters get parentage by asking wardex_sdk._assembly for it."
    )


# --------------------------------------------------------------------------
# C-S2 / C-S3 — one parentage source, one span constructor
# --------------------------------------------------------------------------

# C-S2 has no budget any more. All six parentage sites route
# through `assembly.resolve_parentage()`, which drove the five pre-existing
# `TraceId.generate()` calls (`_tracing.py` 1, `_adapters/_assembler.py` 1,
# `_interceptors/_mcp_stdio.py` 1, `_interceptors/_seam.py` 2) to zero. Per this
# file's header that is the moment the budget is DELETED and the rule becomes
# hard: I1 is now literally true — one call site in the whole SDK — and the
# assertion below says so directly instead of tolerating a count.

# C-S3 has no budget any more either. All six emit sites route
# through `assembly.SpanDraft`, which drove the four pre-existing
# `InternalSpan(...)` construction sites (`_tracing.py` 1,
# `_adapters/_assembler.py` 4, `_interceptors/_mcp_stdio.py` 1,
# `_interceptors/_seam.py` 2 — nine calls across four files) to zero, and took
# every `parent_span_id=` keyword outside `_assembly/` with them: a draft is
# built FROM a `Parentage` and fills the field itself. Per this file's header
# that is the moment the budget is DELETED and the rule becomes hard.


def test_trace_id_is_generated_only_in_parentage():
    """C-S2, now a HARD RULE: exactly one TraceId.generate() in the whole SDK."""
    everywhere = _tally(_calls_traceid_generate)
    assert everywhere == {"_assembly/_parentage.py": 1}, (
        f"C-S2: TraceId.generate() call sites are {everywhere}, expected exactly\n"
        "{'_assembly/_parentage.py': 1}.\n\n"
        "WHY: a new trace id is the statement 'this work has no parent'. One\n"
        "call site is what makes that statement auditable — and it is why the\n"
        "'started a new trace' case can be told apart from the 'expected a\n"
        "parent and lost it' case at all (design I1, I4). Every extra generator\n"
        "is another place a subtree can silently detach into its own trace.\n"
        "This was a budget over five pre-existing sites until they were\n"
        "routed through assembly.resolve_parentage(); it is not a budget\n"
        "any more, so a new site here is a change to make, not a number to\n"
        "raise."
    )


def test_internal_span_is_constructed_in_one_place():
    """C-S3, now a HARD RULE: exactly one InternalSpan(...) in the whole SDK."""
    everywhere = _tally(_constructs_internal_span)
    assert everywhere == {"_assembly/_builder.py": 1}, (
        f"C-S3: InternalSpan is constructed at {everywhere}, expected exactly\n"
        "{'_assembly/_builder.py': 1}.\n\n"
        "WHY: one constructor is what keeps every span carrying correlation,\n"
        "capture_sources and capture_integrity. The two execute_tool variants\n"
        "this SDK shipped differed precisely because they were built in two\n"
        "places — one had all three forensic fields and the other had none —\n"
        "and the span literally named 'chat None' existed because a name was\n"
        "interpolated at an emit site instead of built by the grammar.\n"
        "This was a budget over nine pre-existing calls until they were\n"
        "routed through assembly.SpanDraft; it is not a budget any\n"
        "more, so a new site here is a change to make, not a number to raise."
    )


def test_parent_span_id_is_passed_only_from_assembly():
    """C-S3, second half, also HARD: `parent_span_id=` is _assembly/'s keyword.

    Inside _assembly/ it is unrestricted — that is where the edge is decided.
    """
    outside = {
        k: v for k, v in _tally(_passes_parent_span_id).items() if not k.startswith("_assembly/")
    }
    assert outside == {}, (
        f"C-S3: parent_span_id= is passed outside _assembly/ at {outside}.\n\n"
        "WHY: a parent edge written outside _assembly/ is an edge nobody\n"
        "resolved: no Evidence, no confidence, no limitation marker when it was\n"
        "a guess (design I1, I4). Ask assembly.resolve_parentage() for a\n"
        "Parentage and hand it to SpanDraft, which fills the field itself."
    )


# Hand-built `Ambient(...)` outside _assembly/. Routing the six sites through
# `resolve_parentage()` opened this surface, and it needs the same ratchet as
# the ones that routing closed: `latch_ambient()`
# reads the scope, and a hand-built Ambient is the one way to feed
# `resolve_parentage` a SpanContext that never came from the scope at all —
# `Ambient(SpanContext(trace_id=TraceId(run_id[:16]), ...), ...)` is a framework
# id becoming a parent, which is I2 exactly. C-S2 does not see it (no
# `TraceId.generate()`), C-S3 does not see it (the `parent_span_id=` still comes
# off the returned Parentage), and C-S1 covers only _adapters/.
#
# The one entry is legitimate and is why this is a budget rather than a hard
# rule: `_seam._latched` wraps what `_trackers.py` latched at REQUEST time, and
# the emit path it serves runs on the response side where `latch_ambient()`
# would read the wrong scope. Widening that latch to a real Ambient belongs to
# the seam decomposition (design §3.3), which empties this dict.
_AMBIENT_BUDGET = {
    "_interceptors/_seam.py": 1,
}


def test_ambient_is_latched_not_hand_built():
    _assert_within_budget(
        {k: v for k, v in _tally(_constructs_ambient).items() if not k.startswith("_assembly/")},
        _AMBIENT_BUDGET,
        "C-S2 (Ambient constructed outside _assembly/)",
        "an Ambient is a snapshot of the wardex scope. Building one by hand is\n"
        "the only remaining way to hand resolve_parentage a parent the scope\n"
        "never held — a framework id dressed as a SpanContext (design I2).\n"
        "Call assembly.latch_ambient() on the task that ISSUES the work.",
    )


def test_the_observed_edge_is_told_whether_its_parent_died():
    """design §10.3(b) — every OBSERVING site declares whether its parent had
    already closed.

    `resolve_observed` exists for sites that latch a parent they did not open
    and cannot vet. One of the things they cannot vet is whether the unit behind
    that parent is still running: a `close()` on another carrier leaves the
    finished unit's span installed, and the seam reads it back as
    `contextvar` / 1.0 / no marker into a span that has already shipped.

    `_parentage` cannot ask for itself — `_units` imports it, not the other way
    round — so the fact arrives as a declared argument, the same shape
    `degraded` has. A defaulted argument is exactly the kind of mechanism that
    goes quietly dead when a fourth call site is written, so the rule is
    mechanical rather than a docstring: outside `_assembly/`, there is no
    `resolve_observed(...)` that has not been told.

    Call `assembly.parent_is_closed_unit(parent)` on the task that ISSUES the
    work and carry the answer to the emit path beside the parent itself.
    """
    everywhere = {
        k: v
        for k, v in _tally(_resolves_observed_without_asking).items()
        if not k.startswith("_assembly/")
    }
    assert everywhere == {}, (
        f"resolve_observed is called without `parent_closed=` at {everywhere}.\n\n"
        "WHY: an observed edge whose parent's unit had already closed adopts\n"
        "unrelated later work into an ALREADY-SHIPPED span at confidence 1.0\n"
        "with no marker — one trace where two belong, and the one shape no\n"
        "consumer can detect downstream (design §10.3).\n"
        "Latch assembly.parent_is_closed_unit(parent) on the task that ISSUES\n"
        "the work, store it beside the parent, and declare it here."
    )


# `parent_evicted` is the second defaulted argument on the same function with
# the same failure mode — omit it and the mis-rooted span the argument exists to
# prevent ships silently — so it needs the same mechanical protection. The rule
# cannot be the blanket one above, because three sites legitimately omit it, so
# it is an exact set instead: a NEW omitting call site changes this dict and
# turns red, which is the moment the author has to decide rather than default.
#
# Each entry is a site that cannot see an evicted latch, and the reasons differ:
#   * `_tracing.py` — a HAND-WRITTEN span. It latches the live scope itself at
#     the instant it is called; there is no per-connection table between the
#     latch and the resolve, so nothing can have evicted anything.
#   * `_interceptors/_mcp_stdio.py` — a subprocess pipe. Its pending table is
#     keyed by JSON-RPC id and capped separately, and a dropped pending entry
#     produces no span at all rather than an unparented one.
#   * `_interceptors/_seam.py` — `_build_ws_span`. A WS session inherits its
#     parent from the upgrade transaction, and no cap sits between the two.
# The h2 emit path (`_build_span`, same file) is the one that CAN, which is why
# `_interceptors/_seam.py` appears here with a count of 1 and not 2.
_OBSERVED_WITHOUT_EVICTION = {
    "_tracing.py": 1,
    "_interceptors/_mcp_stdio.py": 1,
    "_interceptors/_seam.py": 1,
}


def test_an_emit_path_that_can_lose_a_latch_entry_declares_it():
    everywhere = {
        k: v
        for k, v in _tally(_resolves_observed_without_declaring_eviction).items()
        if not k.startswith("_assembly/")
    }
    assert everywhere == _OBSERVED_WITHOUT_EVICTION, (
        f"resolve_observed is called without `parent_evicted=` at {everywhere},\n"
        f"expected exactly {_OBSERVED_WITHOUT_EVICTION}.\n\n"
        "WHY: a seam that latches a parent into a CAPPED per-connection table\n"
        "can lose it to its own bound before the response claims it. Omitting\n"
        "the argument ships that span as an honest trace root at confidence\n"
        "1.0 — wardex's own defect presented as a fact about the traffic, and\n"
        "under the default capture mode the gate drops it before anything can\n"
        "say otherwise (_interceptors/_trackers.py, the h2 stream latch).\n"
        "If your site has no such table, add it above WITH THE REASON. If it\n"
        "has one, carry the eviction beside the parent and declare it here."
    )


# --------------------------------------------------------------------------
# design §4.4 / §5.1 — one capture gate (HARD RULES, never budgets)
# --------------------------------------------------------------------------
#
# Both of these are hard from the day they land, and deliberately so. There was
# never a budget to ratchet here: the drift they closed was not N copies of a
# helper, it was TWO implementations of one predicate that had silently grown
# apart (`ByteSeamInterceptor._should_capture` and the `RawSocketInterceptor`
# override that replaced it). A budget of 2 would have been a licence to keep
# them.


def test_the_capture_mode_is_read_in_one_place():
    """The configured policy is consulted only by the module that owns it."""
    everywhere = _tally(_reads_capture_mode)
    assert everywhere == {"_assembly/_policy.py": 1}, (
        f"capture_mode is read at {everywhere}, expected exactly\n"
        "{'_assembly/_policy.py': 1}.\n\n"
        "WHY: a second reader is a second policy. That is not hypothetical —\n"
        "`_interceptors/_socket.py` overrode the gate without ever reading the\n"
        "mode, so `capture_mode=ALL` did nothing on the plaintext seam and a\n"
        "plaintext request inside a live wardex span was dropped while the same\n"
        "bytes over TLS were kept. Neither was decided by anyone; both are what\n"
        "'the same rule, written twice' looks like later (design §4.4).\n"
        "A seam with an opinion about a CONNECTION returns a Prefilter; a seam\n"
        "with an opinion about the POLICY is a bug being written."
    )


def test_the_capture_predicate_has_one_implementation_and_one_composition():
    everywhere = _tally(_defines_a_capture_predicate)
    assert everywhere == {"_assembly/_policy.py": 1, "_interceptors/_seam.py": 1}, (
        f"capture predicates are defined at {everywhere}, expected exactly\n"
        "{'_assembly/_policy.py': 1, '_interceptors/_seam.py': 1}.\n\n"
        "WHY: those two are different jobs and the rule names both so that\n"
        "neither can quietly become the other. `assembly._policy.should_capture`\n"
        "IS the policy. `ByteSeamInterceptor._should_capture` composes it with\n"
        "the seam's `_transport_prefilter` and fails open around the semantic\n"
        "parse — it decides nothing itself (design §4.4).\n"
        "A third entry is an override, and an override is how this diverged the\n"
        "first time. Override `_transport_prefilter` instead: that is the hook\n"
        "for 'this seam knows something about this connection'."
    )


# --------------------------------------------------------------------------
# C-S4 — swallow, but never in silence
# --------------------------------------------------------------------------

# design §10.4 scopes C-S4 to _adapters/ and _interceptors/; _assembly/ is held to
# zero from day one. (`_runtime.py`, `context/_inject.py`, `context/_asgi.py`,
# `context/_propagate.py` and `context/_wsgi.py` also swallow silently today and
# are out of the rule's declared scope, so they are not counted below.)
#
# These numbers went UP once, on the commit that widened `_silent_swallow_node`
# from "the body is literally `pass`" to "the handler leaves no trace" (44 -> 57).
# That is the one legitimate reason a budget may rise: the code did not regress,
# the predicate stopped being blind. Reading it as licence to raise a number is
# the misuse this file's header forbids — from here they only go down.
_CS4_BUDGET = {
    "_adapters/__init__.py": 1,
    # 14 -> 2. Twelve of them were the `try/except Exception: pass`
    # pairs around every tee callback, every patched entry point and the tool
    # wrapper; they are `guard()` blocks now, so the same swallow is counted and
    # logged under `config.debug` instead of being indistinguishable from wardex
    # not being installed. The two that remain are absences rather than failures:
    # `import claude_agent_sdk` (the package is not installed — the answer to the
    # question install() asks) and `asyncio.current_task()` outside a loop, which
    # is how the stdlib spells "the carrier here is the thread".
    "_adapters/_anthropic_agent_sdk.py": 2,
    "_adapters/_assembler.py": 2,
    # Two, and the same justification as the adapter above: both are ABSENCES
    # rather than failures. `_import_pregel` asks "is langgraph installed" and
    # `_import_toolnode` asks "is langgraph-prebuilt installed" — a separately
    # versioned distribution that can be missing on its own — and an ImportError
    # is the ANSWER to each, returned as `None` and branched on by `install()`.
    # Neither can be a module-level import: that would make the decline path
    # dead, surfacing the error on `_adapters/__init__.py`'s stderr line instead
    # of declining silently.
    "_adapters/_langgraph.py": 2,
    "_interceptors/_conn_timing.py": 10,
    # 13 -> 7. The six that went are the ones the patch mechanism made
    # unnecessary: two `except Exception: self._orig_* = None` around install,
    # two `except Exception: pass` around the uninstall `setattr`s (all four
    # replaced by `PatchSet`, whose restore is guarded per patch), and the two
    # fail-silent wrappers around `_wrap_proc`/`_wrap_asyncio_proc`, which are
    # now `guard()` blocks. Lowered in the same commit rather than left stale:
    # `_assert_within_budget` only fails on `actual > budget`, so a number left
    # high is a free slot for a brand-new silent swallow that no test notices.
    "_interceptors/_mcp_stdio.py": 7,
    # 5 -> 4, and the line below is where the fifth went. Extracting
    # `build_grpc_fields` into `_semantics/` took its `except Exception:` with it.
    # Leaving this at 5 would have handed the seam a free slot for a BRAND NEW
    # silent swallow that no test would notice, because `_assert_within_budget`
    # only fails on `actual > budget` — a stale-high number passes in silence.
    # The pair of edits records a transfer; a lone decrement would be a discount
    # for work nobody did.
    "_interceptors/_seam.py": 4,
    "_interceptors/_socket.py": 6,
    "_interceptors/_ssl.py": 6,
    "_semantics/_grpc.py": 1,
}


# C-S4's second spelling, and the only place `_assembly/` is allowed one. A
# `contextlib.suppress` is a silent swallow wherever it appears — it cannot
# report, for the reason `_silent_swallow_node` gives — so the ones that are
# nonetheless CORRECT are listed here by name instead of being exempted by
# predicate. Keyed by module, naming the exceptions each block suppresses, one
# entry per occurrence.
#
# Asserted with `==` and never with a subset test, which is the difference
# between a record and a loophole: a table that only had to be a superset would
# let a SECOND suppress land in a module already listed and change nothing that
# any test can see, and that arrival is the entire reason this table exists.
# It may only shrink.
_ASSEMBLY_SUPPRESS: dict[str, tuple[str, ...]] = {
    # `_current_task()` asks "is there a running event loop" and RETURNS the
    # answer — the thread, when there is not. `asyncio.current_task()` is the
    # only public spelling of that question and reports "no loop" by raising, so
    # the RuntimeError is the answer rather than a failure, and nothing is being
    # hidden from anyone. A counter here would fire on every synchronous
    # `activate()` and be noise, not evidence.
    "_assembly/_units.py": ("RuntimeError",),
}


def _suppressed_exceptions(under: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """{module path: what each `contextlib.suppress` under `under` suppresses}.

    One entry per OCCURRENCE, ordered by line, so two blocks suppressing the
    same class read as two entries rather than collapsing into one. The entry is
    the argument list as written (`"RuntimeError"`, `"OSError, ValueError"`),
    because widening an existing block from `RuntimeError` to `Exception` is the
    change that turns a sanctioned answer into a swallowed failure, and a count
    alone would not see it.
    """
    out: dict[str, tuple[str, ...]] = {}
    for rel, tree in _modules().items():
        if not rel.startswith(under):
            continue
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            for call in _suppress_items(node):
                found.append((call.lineno, ", ".join(ast.unparse(a) for a in call.args)))
        if found:
            out[rel] = tuple(spelling for _, spelling in sorted(found))
    return out


def test_no_silent_swallow_in_assembly():
    sanctioned = _suppressed_exceptions(under=("_assembly/",))
    assert sanctioned == _ASSEMBLY_SUPPRESS, (
        f"the `contextlib.suppress` blocks in _assembly/ are {sanctioned},\n"
        f"and the recorded set is {_ASSEMBLY_SUPPRESS}.\n\n"
        "WHY: `suppress` is the one silent-swallow spelling that can never\n"
        "report itself — its body runs on the SUCCESS path, so a counter written\n"
        "inside it says nothing about the exception that unwound past it. A new\n"
        "one is only correct when the raise is an ANSWER the code goes on to\n"
        "return, never when it is a failure. If that is what you have, add it\n"
        "here with the reason; if it is not, use assembly._diag.guard()."
    )

    inside = _tally(_is_silent_swallow, under=("_assembly/",))
    assert inside == {rel: len(v) for rel, v in _ASSEMBLY_SUPPRESS.items()}, (
        f"C-S4: silent swallow inside _assembly/ at {inside}.\n\n"
        "WHY: _assembly/ owns the ONE authorized swallow — assembly._diag.guard(),\n"
        "which always counts and logs with a traceback under config.debug. A\n"
        "bare `except Exception: pass` in the module that defines the rule turns\n"
        "an SDK bug into 'wardex just doesn't capture this' with no evidence\n"
        "anywhere (design I6).\n"
        "The expected counts are the `_ASSEMBLY_SUPPRESS` entries and nothing\n"
        "else, so a handler added ALONGSIDE a recorded suppress still fires this."
    )


def test_silent_swallows_do_not_spread():
    # `_semantics/` joined the scope the moment the package existed. A rule whose
    # scope is a list of directories goes blind the instant a refactor creates a
    # new one, and the failure is invisible: the suite gets GREENER, because the
    # tallied total drops by whatever moved out of scope. That is the same shape
    # as raising a budget, arrived at without anyone typing a number.
    _assert_within_budget(
        _tally(_is_silent_swallow, under=("_adapters/", "_interceptors/", "_semantics/")),
        _CS4_BUDGET,
        "C-S4 (silent swallow in _adapters/ or _interceptors/)",
        "wardex must not raise into the host, but a swallow that leaves no\n"
        "counter and no debug traceback is indistinguishable from wardex not\n"
        "being installed. Wrap the block in assembly._diag.guard(where=...)\n"
        "instead; the ones already here are converted as they are touched.",
    )


# --------------------------------------------------------------------------
# C-S5 — one sink
# --------------------------------------------------------------------------

_CS5_BUDGET = {
    # Was `__init__.py: 1` until `capture_state_snapshot`'s body moved WHOLE
    # into `_snapshot_api.py` so the package root stops importing `_assembly`.
    # A relocation, not a new occurrence — the total is unchanged, the same
    # shape as `_adapters/_sink.py` below.
    "_snapshot_api.py": 1,
    "_tracing.py": 1,
    # 4 -> 3 + 1: the assembler's `_ClientSink` moved WHOLE into `_adapters/_sink.py`.
    # A relocation, not a new occurrence — the total is unchanged and the
    # assembler's own budget ratchets down, which is the only direction this
    # table allows. The new entry is the last one that will need lowering: this
    # rule's stated destination is `_assembly/_emit.py`, and the sink standing
    # alone in a module of its own is what lets it arrive there whole. When it
    # does, the entry does not shrink — it disappears.
    "_adapters/_assembler.py": 3,
    "_adapters/_sink.py": 1,
    "_interceptors/_mcp_stdio.py": 2,
    "_interceptors/_seam.py": 2,
}


def test_sink_is_not_called_from_assembly_yet():
    inside = _tally(_calls_sink, under=("_assembly/",))
    assert inside == {}, (
        f"C-S5: _assembly/ calls the client sink at {inside}. When it does, it is\n"
        "from _assembly/_emit.py and nowhere else.\n\n"
        "WHY: one sink is where the capture-mode gate, the limitation markers\n"
        "and 'never hold a wardex lock while emitting' (I11) can be enforced\n"
        "once instead of per caller."
    )
    _assert_within_budget(
        _tally(_calls_sink),
        _CS5_BUDGET,
        "C-S5 (capture_span/capture_snapshot outside _assembly/_emit.py)",
        "every direct sink call is a span that skipped the gate and the shared\n"
        "policy, and a place where capture_span can be reached while holding an\n"
        "SDK lock — the reentrancy hazard I11 names.",
    )


# --------------------------------------------------------------------------
# C-S6 — nothing that can raise lives outside the boundary
# --------------------------------------------------------------------------
#
# `ctx.enter()` contains its own failures, which is what lets the host's own
# call sit inside its `with` body. Two places in that statement are NOT
# contained, and neither is obvious:
#
#   * the HEADER. Every argument is evaluated before `__enter__` runs, so a
#     framework attribute read there — `handle.effective_token` on something the
#     SDK moved between releases — breaks the host exactly as it would in the
#     body, and no guard wardex can add will ever see it.
#   * the BODY. Adapter glue belongs in `describe=`, which runs inside the open's
#     own boundary; the same statement written here breaks the host and ships a
#     fabricated ERROR span for a call that never ran.
#
# Both are shape rules because neither is a property anything else can check: a
# call-graph rule cannot see an attribute read, and there is no runtime moment
# at which "this expression was in a header" is observable.

_ENTER_METHODS = frozenset({"enter", "rejoin", "open_run"})


def _enter_calls(node: ast.stmt) -> list[ast.Call]:
    """The `ctx.enter(...)`-family calls a `with` statement opens."""
    if not isinstance(node, ast.With | ast.AsyncWith):
        return []
    out = []
    for item in node.items:
        call = item.context_expr
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _ENTER_METHODS
        ):
            out.append(call)
    return out


def _is_total_expr(node: ast.expr) -> bool:
    """Can this expression be evaluated without running code that might raise?

    A bare name, a literal, an enum member (`UnitKind.CALL` — an attribute on a
    CAPITALIZED name, which is the only attribute read here that cannot be a
    framework object), or `partial(<name>, <total>...)`.

    An attribute read on a lowercase name is exactly the hazard and is refused
    even when the object is wardex's own: the rule cannot tell `handle.tools`
    from `handle.effective_token`, and a rule that had to would go blind the
    first time somebody wrapped one in a property.
    """
    if isinstance(node, ast.Name | ast.Constant):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id[:1].isupper()
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name != "partial":
            return False
        return all(_is_total_expr(a) for a in node.args) and not node.keywords
    return False


def _header_violations(tree: ast.AST) -> list[int]:
    bad: list[int] = []
    for node in ast.walk(tree):
        for call in _enter_calls(node):
            args = list(call.args) + [kw.value for kw in call.keywords]
            if not all(_is_total_expr(a) for a in args):
                bad.append(call.lineno)
    return bad


def _is_pass_through_aiter(stmt: ast.stmt) -> bool:
    """`async for x in <call>: yield x` — the async twin of `yield from <call>`.

    An async generator cannot delegate with `yield from` (it is a syntax error),
    so this loop is the only way to hold a scope across a framework's async
    iteration. The sync entry gets there through the `Return` branch below —
    `return (yield from original(...))` — and the async entry had no legal
    spelling at all; hoisting the `with` into a helper generator does not help,
    because this rule is per-`With` node.

    Admitted in EXACTLY this shape: one `AsyncFor` with no `orelse`, over a
    CALL, one statement inside it, and that statement a bare `yield` of the loop
    variable. Nothing can be COMPUTED in it, which is the whole of what the rule
    is about — `yield _shape(chunk)` and a second statement in the loop both stay
    flagged. The sync twin `out = original(...)` is deliberately NOT admitted: it
    would open the body to arbitrary adapter glue.
    """
    if not isinstance(stmt, ast.AsyncFor) or stmt.orelse:
        return False
    if not isinstance(stmt.target, ast.Name) or not isinstance(stmt.iter, ast.Call):
        return False
    if len(stmt.body) != 1:
        return False
    inner = stmt.body[0]
    return (
        isinstance(inner, ast.Expr)
        and isinstance(inner.value, ast.Yield)
        and isinstance(inner.value.value, ast.Name)
        and inner.value.value.id == stmt.target.id
    )


def _body_violations(tree: ast.AST) -> list[int]:
    """Statements in a `with ctx.enter(...)` body that are neither the host's
    call nor a method call on the yielded scope."""
    bad: list[int] = []
    for node in ast.walk(tree):
        for _call in _enter_calls(node):
            assert isinstance(node, ast.With | ast.AsyncWith)
            yielded = {
                item.optional_vars.id
                for item in node.items
                if isinstance(item.optional_vars, ast.Name)
            }
            for stmt in node.body:
                if isinstance(stmt, ast.Return | ast.Pass):
                    continue
                if _is_pass_through_aiter(stmt):
                    continue  # delegating the host's async iteration IS the host's work
                if isinstance(stmt, ast.Assign | ast.AnnAssign | ast.Expr):
                    value = stmt.value
                    if value is None:
                        continue
                    if isinstance(value, ast.Await):
                        continue  # the host's own call
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id in yielded
                    ):
                        continue  # a verb on the scope, which is total
                    bad.append(stmt.lineno)
                    continue
                bad.append(stmt.lineno)
            break
    return bad


def test_nothing_that_can_raise_lives_in_an_enter_header():
    found = {rel: lines for rel, tree in _modules().items() if (lines := _header_violations(tree))}
    assert found == {}, (
        f"C-S6: a `ctx.enter(...)` header holds an expression that can raise, at {found}.\n\n"
        "WHY: every argument is evaluated BEFORE `__enter__`, so it is outside\n"
        "every failure boundary wardex has — a framework read there breaks the\n"
        "host and no guard can ever see it. Names, literals, enum members and a\n"
        "`partial` of a name are the whole vocabulary. If a site needs a computed\n"
        "value, it belongs in `describe=`, which runs inside the open's boundary."
    )


def test_nothing_but_the_hosts_call_lives_in_an_enter_body():
    found = {rel: lines for rel, tree in _modules().items() if (lines := _body_violations(tree))}
    assert found == {}, (
        f"C-S6: a `ctx.enter(...)` body holds adapter glue, at {found}.\n\n"
        "WHY: the body is the one place the host's own call may live, and it is\n"
        "unguarded on purpose — a guard there would swallow the host's exception\n"
        "and report a failing call as a successful one. Anything else written\n"
        "there breaks the host for a span attribute. Adapter glue goes in\n"
        "`describe=`, which runs inside the same boundary as the open.\n\n"
        "DELEGATION COUNTS AS THE HOST'S CALL: `return (yield from original(...))`\n"
        "and its async twin `async for x in original(...): yield x` are pumping the\n"
        "host's own generator, which is the host's own work. The async form is\n"
        "admitted in that exact shape only — see `_is_pass_through_aiter`."
    )


@pytest.mark.parametrize(
    "source",
    [
        # the hazard the inversion created: a framework read in the header
        "with ctx.enter(K, selector=UnitKey('a', handle.effective_token)) as call:\n    pass\n",
        # an f-string, which is a call in disguise
        "with ctx.enter(K, subject=f'{handle.token}/{name}') as call:\n    pass\n",
        # a helper, however innocent it looks
        "with ctx.enter(K, selector=_call_key(handle, name)) as call:\n    pass\n",
        # a bare attribute read on an adapter-held object
        "with ctx.enter(K, subject=handle.name) as call:\n    pass\n",
        # `partial` of something that is itself computed
        "with ctx.enter(K, describe=partial(f, adapter._names.catalog())) as call:\n    pass\n",
    ],
)
def test_c_s6_sees_a_header_expression_that_can_raise(source):
    assert _header_violations(ast.parse(source)), f"C-S6 went blind on:\n{source}"


@pytest.mark.parametrize(
    "source",
    [
        "with ctx.enter(UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL,\n"
        "               subject=tool_name, fallback=Fallback.SOLE_LIVE_RUN,\n"
        "               describe=partial(_describe, adapter, handle, tool_name, args)) as call:\n"
        "    result = await handler(args)\n"
        "    call.record_output(_tool_input(result))\n",
        # the SYNC run entry: delegating a generator through the `Return` branch
        "with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_WORKFLOW,\n"
        "               placement=Placement.ROOT, describe=partial(_d, a, s)) as run:\n"
        "    return (yield from original(self, *args, **kwargs))\n",
        # the ASYNC run entry: the only spelling an async generator has, since
        # `yield from` is a syntax error in one. Case D.
        "with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_WORKFLOW,\n"
        "               placement=Placement.ROOT, describe=partial(_d, a, s)) as run:\n"
        "    async for chunk in original(self, *args, **kwargs):\n"
        "        yield chunk\n",
    ],
)
def test_c_s6_accepts_the_shape_the_adapter_actually_writes(source):
    tree = ast.parse(source)
    assert _header_violations(tree) == []
    assert _body_violations(tree) == []


@pytest.mark.parametrize(
    "source",
    [
        # adapter glue in the body: breaks the host, and ships a fabricated
        # ERROR span for a call that never ran
        "with ctx.enter(K) as call:\n    call.draft.set_extra('k', adapter._names.token())\n"
        "    result = await handler(args)\n",
        # a bare attribute read, which a call-graph rule cannot see
        "with ctx.enter(K) as call:\n    x = task.name\n    result = await handler(args)\n",
        # a guard nested in the body — there is nothing left for it to protect,
        # and reaching for one is how the host's exception gets swallowed
        "with ctx.enter(K) as call:\n    with adapter._guard('x'):\n        pass\n",
        # --- near-misses of the async delegation shape. Each is one edit away
        # --- from the admitted form, and each would reopen the body to glue.
        # a COMPUTED yield: the loop is now a transform, not a pass-through
        "with ctx.enter(K) as run:\n    async for chunk in original(self):\n"
        "        yield _shape(chunk)\n",
        # a second statement in the loop
        "with ctx.enter(K) as run:\n    async for chunk in original(self):\n"
        "        run.note(M)\n        yield chunk\n",
        # iterating an ATTRIBUTE rather than a call — not a delegation
        "with ctx.enter(K) as run:\n    async for chunk in self.stream:\n        yield chunk\n",
        # the SYNC twin, deliberately not admitted: `yield from` is the sync form
        "with ctx.enter(K) as run:\n    for chunk in original(self):\n        yield chunk\n",
        # an `orelse` runs adapter code after the host's iteration
        "with ctx.enter(K) as run:\n    async for chunk in original(self):\n        yield chunk\n"
        "    else:\n        run.note(M)\n",
        # yielding a DIFFERENT name — the loop variable is not what is passed on
        "with ctx.enter(K) as run:\n    async for chunk in original(self):\n        yield other\n",
    ],
)
def test_c_s6_sees_a_body_that_is_not_only_the_hosts_call(source):
    assert _body_violations(ast.parse(source)), f"C-S6 went blind on:\n{source}"


# --------------------------------------------------------------------------
# C-S7 — a guarded step is tested by a flag, never by what it assigns
# --------------------------------------------------------------------------
#
# The most valuable rule here, because it caught a live bug in a design that was
# about to ship — twice, written by two authors an hour apart. `guard()` swallows
# a failure and execution continues at the statement after the block, so the code
# has to ask "did that work?". The obvious spelling asks a variable the block
# ASSIGNS:
#
#     with self.guard("activate"):
#         activation = unit.activate()      # <- binds
#         activation.__enter__()            # <- raises
#     if activation is None:                # <- DEAD. It is already bound.
#
# The branch never runs, so the degradation is never recorded and the span ships
# byte-identical to a healthy one. The only spelling that is not dead is a bare
# flag set False before the block and True as its LAST statement, because that
# assignment is the one thing a failure anywhere in the block prevents.


def _is_guard_with(node: ast.stmt) -> bool:
    if not isinstance(node, ast.With):
        return False
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "guard":
            return True
    return False


def _names_assigned_in(body: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Assign):
                out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(
                node.target, ast.Name
            ):
                out.add(node.target.id)
    return out


def _assigns_name(stmt: ast.stmt | None, name: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == name
    )


def _assigns_sentinel(stmt: ast.stmt | None, name: str) -> bool:
    """`name = False` or `name = None` — a value the guarded block will replace."""
    return (
        _assigns_name(stmt, name)
        and isinstance(stmt.value, ast.Constant)  # type: ignore[union-attr]
        and stmt.value.value in (None, False)  # type: ignore[union-attr]
    )


def _flag_violations(tree: ast.AST) -> list[int]:
    """Line numbers where a guarded step is tested by a name a failure can leave set.

    Scope is the `if` IMMEDIATELY following the `with`, in the SAME body,
    testing a name the block assigns. What makes a test sound is not its
    SYNTAX — `if not ok:` and `if client is not None:` are equally fine — it is
    WHERE the assignment sits:

      * the name is set to a sentinel (`False` / `None`) immediately BEFORE the
        block, so its pre-block value says "this did not happen"; and
      * the block's LAST statement is the only one that assigns it, so no
        failure inside the block can leave it holding anything else.

    A name assigned by a non-final statement is the trap: it is already bound
    when a later statement raises, so the test is dead and the failure ships
    looking like a success.
    """
    bad: list[int] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if not _is_guard_with(stmt) or i + 1 >= len(body):
                    continue
                nxt = body[i + 1]
                if not isinstance(nxt, ast.If):
                    continue
                tested = {n.id for n in ast.walk(nxt.test) if isinstance(n, ast.Name)}
                before = body[i - 1] if i else None
                for name in sorted(tested & _names_assigned_in(stmt.body)):
                    sound = (
                        _assigns_sentinel(before, name)
                        and _assigns_name(stmt.body[-1], name)
                        and name not in _names_assigned_in(stmt.body[:-1])
                    )
                    if not sound:
                        bad.append(nxt.lineno)
                        break
    return bad


def test_a_guarded_step_is_tested_by_a_flag_and_never_by_what_it_assigns():
    """Package-wide, and not a budget. There is no legitimate instance of this."""
    found = {rel: lines for rel, tree in _modules().items() if (lines := _flag_violations(tree))}
    assert found == {}, (
        f"C-S7: a guarded step is tested by a name a failure can leave set, at {found}.\n\n"
        "WHY: `guard()` swallows, so the statement after the block has to ask\n"
        "whether the block finished. A name assigned by a non-final statement is\n"
        "already bound when a later one raises, so the test is dead code and the\n"
        "failure ships looking like a success. Write:\n\n"
        "    ok = False\n"
        "    with self.guard(...):\n"
        "        ...\n"
        "        ok = True\n"
        "    if not ok:\n\n"
        "The `None` sentinel is equally fine when the block's LAST statement is\n"
        "the only one that assigns the name."
    )


@pytest.mark.parametrize(
    "source",
    [
        # objection 8's exact bug, and the one that shipped in a design review
        "def f(self):\n"
        "    activation = None\n"
        "    with self.guard('a'):\n"
        "        activation = unit.activate()\n"
        "        activation.__enter__()\n"
        "    if activation is None:\n"
        "        degrade()\n",
        # the same trap wearing a truthiness test
        "def f(self):\n"
        "    scope = None\n"
        "    with self.guard('a'):\n"
        "        scope = Scope(unit, self)\n"
        "        describe(scope)\n"
        "    if not scope:\n"
        "        degrade()\n",
        # a flag, but never initialised to False before the block
        "def f(self):\n"
        "    with self.guard('a'):\n"
        "        work()\n"
        "        ok = True\n"
        "    if not ok:\n"
        "        degrade()\n",
        # a flag set True too EARLY, so a later failure still reads as success
        "def f(self):\n"
        "    ok = False\n"
        "    with self.guard('a'):\n"
        "        ok = True\n"
        "        work()\n"
        "    if not ok:\n"
        "        degrade()\n",
        # the module-level `guard(...)` spelling, not only `self.guard(...)`
        "def f():\n"
        "    unit = None\n"
        "    with guard('a'):\n"
        "        unit = open()\n"
        "        bind(unit)\n"
        "    if unit is not None:\n"
        "        use(unit)\n",
    ],
)
def test_c_s7_sees_a_dead_test_however_it_is_written(source):
    assert _flag_violations(ast.parse(source)), f"C-S7 went blind on:\n{source}"


@pytest.mark.parametrize(
    "source",
    [
        # the sanctioned shape
        "def f(self):\n"
        "    ok = False\n"
        "    with self.guard('a'):\n"
        "        work()\n"
        "        ok = True\n"
        "    if not ok:\n"
        "        degrade()\n",
        # the `None` sentinel, which is the same protocol with a different value:
        # the block's LAST statement is the only one that assigns the name, so a
        # failure anywhere leaves it None. Three modules already use this shape.
        "def f(self):\n"
        "    span = None\n"
        "    with guard('a'):\n"
        "        span = build(pending, m)\n"
        "    if span is not None:\n"
        "        out.append(span)\n",
        # and with other work before the assignment, which is still sound
        "def f(self):\n"
        "    snapshot = None\n"
        "    with guard('a'):\n"
        "        draft = build()\n"
        "        draft.set_extras(attrs)\n"
        "        snapshot = draft.finish(now)\n"
        "    if snapshot is not None:\n"
        "        client.capture_snapshot(snapshot)\n",
        # an `if` that has nothing to do with the guarded block
        "def f(self):\n"
        "    with self.guard('a'):\n"
        "        work()\n"
        "    if self.debug:\n"
        "        report()\n",
        # a `with` that is not a guard at all
        "def f(self):\n"
        "    unit = None\n"
        "    with self._lock:\n"
        "        unit = open()\n"
        "    if unit is None:\n"
        "        return\n",
    ],
)
def test_c_s7_does_not_flag_a_step_that_is_tested_correctly(source):
    assert _flag_violations(ast.parse(source)) == [], f"C-S7 false-positived on:\n{source}"


# --------------------------------------------------------------------------
# the degraded draft answers everything the real one does
# --------------------------------------------------------------------------


def _public_names(cls: type) -> frozenset[str]:
    return frozenset(n for n in vars(cls) if not n.startswith("_"))


def test_the_degraded_draft_answers_every_verb_the_real_one_does():
    """Reflection, not AST: a verb added to `SpanDraft` needs a null counterpart.

    `NULL_DRAFT` is what an adapter is handed after wardex's own work failed,
    and the whole reason the adapter does not have to branch on that is that
    every verb still answers. A hand-written null draft covers the verbs its
    author happened to remember — which is a list written from the same memory
    that will later add a setter and forget it — so the superset is asserted
    here instead, where the failure lands in this suite rather than in a host's
    process as an `AttributeError` from inside a `with` body.

    Three exclusions, and each is structural rather than convenient:

      * `finish` — a null draft may never reach a sink. The only way to make
        that true by construction is for the materializer not to exist, so its
        ABSENCE is asserted rather than a no-op version being tolerated.
      * `manual` / `transport` — constructors, not verbs. Nothing holding a
        degraded draft calls them; they are how a real one is born.
    """
    from wardex_sdk._assembly._builder import NULL_DRAFT, IntegrityBuilder, SpanDraft

    null = _public_names(type(NULL_DRAFT))
    excluded = {"finish", "manual", "transport"}

    missing = (_public_names(SpanDraft) - excluded) - null
    assert missing == set(), (
        f"NULL_DRAFT cannot answer {sorted(missing)}, which SpanDraft can. An\n"
        "adapter handed a degraded scope keeps describing it — that is the point\n"
        "of not making it branch — so every one of those calls has to land\n"
        "somewhere that cannot fail. Add the no-op to _assembly/_builder.py."
    )
    # Asked of what `.integrity` RETURNS, not of the null draft's own class.
    # `_NullDraft` answers itself there, which is an implementation choice — the
    # obligation is that whatever comes back speaks the builder's language.
    missing_integrity = _public_names(IntegrityBuilder) - _public_names(type(NULL_DRAFT.integrity))
    assert missing_integrity == set(), (
        f"NULL_DRAFT.integrity cannot answer {sorted(missing_integrity)}, which\n"
        "IntegrityBuilder can. `draft.integrity` is reached by every site that\n"
        "needs the full attempted/ok vocabulary rather than set_io's shorthand."
    )

    assert not hasattr(NULL_DRAFT, "finish"), (
        "NULL_DRAFT grew a finish(). A null draft that can be materialized is a\n"
        "degraded span that can be emitted — the absence is the mechanism."
    )


# --------------------------------------------------------------------------
# negative controls — the rules above must FAIL on a real bypass
# --------------------------------------------------------------------------
#
# A structural rule that nobody has watched fail is a rule nobody knows the
# shape of. Every predicate here was, at some point in this file's history,
# green against source that did the exact thing it forbids: `_types.TraceId`
# reached the type without naming it, `except Exception: return None` swallowed
# without naming `pass`. These cases pin the spellings that used to escape, so
# the next widening of a predicate is a test change and not a discovery.


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _handler(body: str) -> str:
    """A function with one `except Exception:` handler whose body is `body`."""
    return f"def f():\n    try:\n        g()\n    except Exception:\n        {body}\n"


_BYPASS_SPELLINGS = {
    "bare symbol": "from .._types import TraceId, InternalSpan",
    "aliased symbol": "from .._types import TraceId as _T, InternalSpan as _S",
    "module attribute": "from .. import _types",
    "aliased module": "from .. import _types as _t",
    "absolute module": "import wardex_sdk._types",
    "package attribute": "import wardex_sdk",
    "from-package module": "from wardex_sdk import _types",
}


def _bypass_source(header: str) -> str:
    """The same bypass written against whatever `header` bound."""
    if "InternalSpan" in header and " as " not in header:
        body = "def f():\n    return TraceId.generate(), InternalSpan(1, 2)\n"
    elif " as _T" in header:
        body = "def f():\n    return _T.generate(), _S(1, 2)\n"
    elif header.endswith("_types as _t"):
        body = "def f():\n    return _t.TraceId.generate(), _t.InternalSpan(1, 2)\n"
    elif header == "import wardex_sdk._types" or header == "import wardex_sdk":
        body = (
            "def f():\n"
            "    return wardex_sdk._types.TraceId.generate(), "
            "wardex_sdk._types.InternalSpan(1, 2)\n"
        )
    else:
        body = "def f():\n    return _types.TraceId.generate(), _types.InternalSpan(1, 2)\n"
    return header + "\n\n\n" + body


@pytest.mark.parametrize("spelling", sorted(_BYPASS_SPELLINGS))
def test_every_spelling_of_the_span_machinery_bypass_is_seen(spelling):
    """C-S2/C-S3: reaching the type counts, however the reach is written."""
    rel = "_adapters/_probe.py"
    tree = _parse(_bypass_source(_BYPASS_SPELLINGS[spelling]))

    traces = [n for n in ast.walk(tree) if _calls_traceid_generate(rel, tree)(n)]
    spans = [n for n in ast.walk(tree) if _constructs_internal_span(rel, tree)(n)]

    assert len(traces) == 1, f"C-S2 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]}"
    assert len(spans) == 1, f"C-S3 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]}"


@pytest.mark.parametrize(
    "spelling", ["module attribute", "absolute module", "package attribute", "from-package module"]
)
def test_c_s1_sees_a_module_import_not_only_a_symbol_import(spelling):
    """C-S1: `from .. import _types` is the bypass, and it names no symbol."""
    rel = "_adapters/_probe.py"
    tree = _parse(_bypass_source(_BYPASS_SPELLINGS[spelling]))

    named = _imported_names(tree) & _FORBIDDEN_IN_ADAPTERS
    modules = _imported_modules(rel, tree) & _FORBIDDEN_MODULES_IN_ADAPTERS

    assert not named, "precondition: this spelling deliberately names no forbidden symbol"
    assert modules, (
        f"C-S1 blind to {spelling!r}: {_BYPASS_SPELLINGS[spelling]} — an adapter\n"
        "can reach TraceId/SpanContext/InternalSpan through the module object."
    )


@pytest.mark.parametrize(
    "body",
    [
        "pass",
        "...",
        "return None",
        "return",
        'return b""',
        "return False",
        "self._disabled = True",
    ],
)
def test_c_s4_sees_a_swallow_that_leaves_no_trace_however_written(body):
    tree = _parse(_handler(body))

    handlers = [n for n in ast.walk(tree) if _silent_swallow_node(n)]

    assert len(handlers) == 1, f"C-S4 blind to a handler whose body is {body!r}"


@pytest.mark.parametrize(
    "body",
    [
        "raise",
        "raise RuntimeError from None",
        "counters.bump('x')",
        "logger.warning('x')",
        "parser_disable_log('x')",
        "with guard('x'):\n            pass",
    ],
)
def test_c_s4_does_not_flag_a_handler_that_reports_itself(body):
    tree = _parse(_handler(body))

    assert not [n for n in ast.walk(tree) if _silent_swallow_node(n)], (
        f"C-S4 false positive on a self-reporting handler: {body!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "import contextlib\n\ndef f():\n    with contextlib.suppress(Exception):\n        g()\n",
        "from contextlib import suppress\n\ndef f():\n    with suppress(Exception):\n        g()\n",
        (
            "import contextlib\n\nasync def f():\n"
            "    async with contextlib.suppress(Exception):\n        await g()\n"
        ),
        # The escape hatch an `except` handler gets does NOT apply here: this
        # body runs on the success path, so the bump proves nothing about the
        # exception that unwound past it.
        (
            "import contextlib\n\ndef f():\n"
            "    with contextlib.suppress(Exception):\n        counters.bump('x')\n"
        ),
    ],
)
def test_c_s4_sees_the_suppress_spelling_of_a_swallow(source):
    tree = _parse(source)

    assert [n for n in ast.walk(tree) if _silent_swallow_node(n)], (
        "C-S4 blind to a `contextlib.suppress` swallow:\n"
        f"{source}\n"
        "A `try/except: pass` and a `with suppress` discard the same exception."
    )


@pytest.mark.parametrize(
    "source",
    [
        # wardex's own CM in `_suppress.py`. It suppresses CAPTURE
        # so the exporter's own HTTP call is not re-captured; it swallows no
        # exception, and reading it as one would put a false entry in the table.
        "def f():\n    with suppress_capture():\n        g()\n",
        "def f():\n    with open('x') as fh:\n        fh.read()\n",
    ],
)
def test_c_s4_does_not_flag_a_with_that_swallows_nothing(source):
    tree = _parse(source)

    assert not [n for n in ast.walk(tree) if _silent_swallow_node(n)], (
        f"C-S4 false positive on a `with` that discards no exception:\n{source}"
    )


@pytest.mark.parametrize(
    "call",
    [
        'importlib.import_module("wardex_sdk._interceptors._seam")',
        '__import__("wardex_sdk._adapters._assembler")',
    ],
)
def test_dynamic_imports_are_resolved_like_static_ones(call):
    tree = _parse(f"import importlib\n\n_m = {call}\n")

    found = _imported_modules("_assembly/_probe.py", tree)

    assert any(
        t.startswith(f"{_PKG}.") and "interceptors" in t or "adapters" in t for t in found
    ), f"a layering rule can be escaped by writing the import as {call}"


def test_a_non_literal_dynamic_import_is_reported_as_unauditable():
    tree = _parse('import importlib\n\n_m = importlib.import_module("wardex_sdk." + name)\n')

    assert "<dynamic>" in _imported_modules("_assembly/_probe.py", tree)


def test_a_re_exported_symbol_from_a_package_is_not_invisible():
    """`from .. import conversation` names no module, and `_tracing.py` is above _assembly/."""
    tree = _parse("from .. import capture_state_snapshot, conversation\n")

    found = _imported_modules("_assembly/_probe.py", tree)

    assert f"{_PKG}:conversation" in found
    assert f"{_PKG}:capture_state_snapshot" in found
