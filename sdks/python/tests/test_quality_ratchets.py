"""The quality budgets, enforced.

A budget is a number recorded here that may move in ONE direction. Each test
below reads the tree and compares against the recorded value. If a test here
fails, the change under review moved a number the wrong way: undo the change,
or lower the recorded value in the same commit. Raising a value here is the
one edit this file is not for — a raise is a commit of its own, subject
`quality:`, whose body says what is being given up and why, so that it is
reviewable on its own and cannot ride along inside an unrelated change.

The failure each ratchet closes:

  * `unwrap()`/`expect(` in non-test Rust — a panic costs the host a record
    or a connection, and its message lands on the host's stderr. The compiler
    is the gate (`clippy::unwrap_used` / `expect_used` in the workspace
    lints); the text scan here is kept identical to the one in
    `scripts/quality-snapshot.sh`, so the number the snapshot prints is a
    number a test also defends.
  * a panic-site counter that stops reading a file at the first `#[cfg(test)]`
    line — the marker does not always open a trailing test module. In
    `crates/wardex-protocol/src/semantic/mod.rs` it decorates the one-line
    declaration `mod tests;` on line 13 of 532, and in
    `crates/wardex-pipeline/src/pii/walk.rs` it decorates a test-only `if`
    in the middle of a live function. Truncating there abandoned a whole
    production file and missed three `.unwrap()` calls in the SSE dispatch,
    which is exactly the code this budget exists to count. A counter that
    under-reports reads precisely like a ratchet holding, so nothing turns
    red while the number decays: `_count_unwrap_lines` skips the
    brace-matched `#[cfg(test)]` ITEM and keeps scanning the rest of the file.
    A file that exists only because a `#[cfg(test)] mod name;` declaration
    pulls it in (`semantic/tests.rs`) is test code too, and is skipped whole;
    counting it once hid 167 lines of slack behind a budget of 172.
  * modules over 800 lines — the file nobody dares open is where the second
    emitter path gets added.
  * hand-run checks missing from `AGENTS.md` — a check nobody is told to run
    is a check nobody runs; four Python files were unlisted the day this file
    was written, and all eight `#[ignore]`d Rust probes were unlisted the day
    the scan widened to see them.
  * a skip, xfail or `importorskip` whose reason is not written on the marker
    itself — a test that stops running without saying why is a test nobody can
    ever decide to re-enable, because nobody can tell whether the cause is
    still real.
  * a runtime Python dependency — a host's resolver fighting ours is the first
    install failure an observability SDK can cause.
  * a public name without a docstring — `help()` is the first documentation a
    host reads, and an empty one says the API is unfinished.

Almost every ratchet here reads source text and never the imported package, so
a stale native wheel cannot make it lie. Two do something else, for two
different reasons.

The skip-reason ratchet parses each test module with `ast`. A marker's reason
can sit many physical lines below the marker that carries it — one `skipif` in
this tree spans fourteen lines — so a line-anchored regex would flag a
decorator whose reason is right there, and would miss a message passed
positionally to `pytest.skip`. `ast` collapses each marker into a single call
node and ends both problems. It keeps the rule above: `ast.parse` reads the
source TEXT and never imports the package, so a stale native wheel still
cannot make it lie.

`test_public_docstring_debt_only_falls` is the other, and it is the one test
here that DOES import the package: a docstring exists only as a runtime
attribute, so no amount of reading text reproduces what `help()` will print.
A stale wheel cannot move that number either, but for a different reason —
every name in `__all__` is defined in Python source in this tree, and the
native extension exports nothing public (its symbols are private and
underscore-aliased), so rebuilding it or failing to rebuild it changes no
docstring the test reads.

When a ratchet reaches zero, delete its recorded value and rewrite the test as
a hard rule (the `test_import_graph.py` BUDGET → HARD RULE convention).
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PY_SRC = _ROOT / "sdks" / "python" / "src" / "wardex_sdk"
_AGENTS = _ROOT / "AGENTS.md"
_PYPROJECT = _ROOT / "sdks" / "python" / "pyproject.toml"
_TESTS = pathlib.Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# P1 — panic surface in Rust
# --------------------------------------------------------------------------

#: `unwrap()`/`expect(` lines in non-test Rust, counted by `_count_unwrap_lines`
#: (same method as `scripts/quality-snapshot.sh`; the two implementations must
#: stay identical, and a divergence between them is its own defect).
#:
#: This was a budget (recorded 2026-09-11 as 169, corrected to 172 on
#: 2026-09-13 when the counter stopped abandoning files at a `mod tests;`
#: line). It became a HARD RULE at zero on 2026-11-08, when the counter also
#: stopped counting files that only a `#[cfg(test)] mod name;` declaration
#: pulls in (167 of the 172 were `semantic/tests.rs`), and the last production
#: sites were rewritten to prove their invariant by pattern. The compiler now
#: enforces the same rule through `clippy::unwrap_used` and `expect_used`.

_UNWRAP_RE = re.compile(r"\.unwrap\(\)|\.expect\(")
_CFG_TEST_RE = re.compile(r"#\[\s*cfg\(test\)\s*\]")

#: A raw string opener (`r"`, `r#"`, `br##"`), a Rust char literal, and an
#: identifier character — the three shapes `_rust_code_lines` has to recognise
#: to blank out a literal without eating the code around it.
_RAW_STR_RE = re.compile(r'b?r(#*)"')
_CHAR_LIT_RE = re.compile(r"'(?:\\.|[^\\'])'")
_IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")


def _blanked(chunk: str) -> str:
    """`chunk` with every character but the newlines replaced by a space."""
    return "".join("\n" if c == "\n" else " " for c in chunk)


def _rust_code_lines(text: str) -> list[str]:
    """`text` split into lines with comments and literals blanked out.

    Line for line with the original, so a caller can match on the raw text and
    count braces on this. Blanking is what makes the brace matching below
    survive this tree: the test modules are full of JSON fixtures, and a `{`
    inside a fixture string or inside a `//!` doc comment is not a block.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and text[i + 1 : i + 2] == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            out.append(_blanked(text[i:end]))
            i = end
            continue
        if ch == "/" and text[i + 1 : i + 2] == "*":
            depth, end = 1, i + 2
            while end < n and depth:
                if text[end : end + 2] == "/*":
                    depth += 1
                    end += 2
                elif text[end : end + 2] == "*/":
                    depth -= 1
                    end += 2
                else:
                    end += 1
            out.append(_blanked(text[i:end]))
            i = end
            continue
        raw = _RAW_STR_RE.match(text, i)
        if raw and not (i and _IDENT_CHAR_RE.match(text[i - 1])):
            close = '"' + raw.group(1)
            end = text.find(close, raw.end())
            end = n if end < 0 else end + len(close)
            out.append(_blanked(text[i:end]))
            i = end
            continue
        if ch == '"':
            end = i + 1
            while end < n:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    end += 1
                    break
                end += 1
            out.append(_blanked(text[i:end]))
            i = end
            continue
        if ch == "'":
            lit = _CHAR_LIT_RE.match(text, i)
            if lit:  # a char literal; a lifetime has no closing quote
                out.append(_blanked(lit.group(0)))
                i = lit.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out).splitlines()


def _end_of_cfg_test_item(code: list[str], start: int, col: int) -> int:
    """Index of the first line after the item a `#[cfg(test)]` decorates.

    `code` is blanked source, `start`/`col` the position just past the marker.
    Two shapes have to end in the right place. A block item (`mod tests {`, or
    a test-only `if` inside a live function) ends where its braces balance. A
    declaration (`mod tests;`, a `const`) has no block at all and ends at its
    semicolon — the case that used to make the scanner abandon a 532-line
    production file at line 13. Bracket depth is tracked only so that the `;`
    in `[u8; 4]`, or a comma-bearing attribute below the marker, cannot end
    the item early.
    """
    brace = bracket = 0
    opened = False
    for j in range(start, len(code)):
        for ch in code[j][col:] if j == start else code[j]:
            if ch == "{":
                brace += 1
                opened = True
            elif ch == "}":
                brace -= 1
            elif ch in "([":
                bracket += 1
            elif ch in ")]":
                bracket -= 1
            elif ch == ";" and not opened and brace == 0 and bracket == 0:
                return j + 1
        if opened and brace <= 0:
            return j + 1
    return len(code)


_CFG_TEST_MOD_DECL_RE = re.compile(
    r"#\[\s*cfg\(test\)\s*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;"
)
#: An `#[expect(clippy::unwrap_used | expect_used, reason = "...")]` marks the
#: one sanctioned shape of a production `expect`: the attribute states the
#: invariant for review, and the compiler reports it the day the site is gone.
#: The item it decorates is skipped the way a `#[cfg(test)]` item is.
_EXPECT_ATTR_OPEN_RE = re.compile(r"#\[\s*expect\s*\(")
_PANIC_LINTS_RE = re.compile(r"\bclippy::(?:unwrap_used|expect_used)\b")


def _expect_panic_marker(code: list[str], i: int) -> tuple[int, int] | None:
    """`(line, col)` just past the closing `]` of an `#[expect(..)]` attribute
    that starts on line `i` of the blanked code and names `unwrap_used` or
    `expect_used`; `None` when line `i` starts no such attribute.

    The attribute is read to its own `]` first, across lines, because rustfmt
    wraps any `reason` past the line width onto lines of its own: judging the
    first line alone would reject the exact shape the error message asks for.
    """
    m = _EXPECT_ATTR_OPEN_RE.search(code[i])
    if m is None:
        return None
    depth = 0
    parts: list[str] = []
    for j in range(i, len(code)):
        line = code[j]
        start = m.start() if j == i else 0
        for k in range(start, len(line)):
            if line[k] == "[":
                depth += 1
            elif line[k] == "]":
                depth -= 1
                if depth == 0:
                    parts.append(line[start : k + 1])
                    return (j, k + 1) if _PANIC_LINTS_RE.search("\n".join(parts)) else None
        parts.append(line[start:])
    return None


def _test_only_files(root: pathlib.Path) -> set[pathlib.Path]:
    """Files that exist only because a `#[cfg(test)] mod name;` pulls them in.

    `mod tests;` under `#[cfg(test)]` makes `tests.rs` (or `tests/mod.rs`)
    beside it a test file whose own text carries no marker at all, so the
    per-item skip below cannot see it. The declaration is matched on blanked
    code so a `mod` in a comment or a string does not count.
    """
    out: set[pathlib.Path] = set()
    for path in root.rglob("*.rs"):
        code = "\n".join(_rust_code_lines(path.read_text(encoding="utf-8")))
        for m in _CFG_TEST_MOD_DECL_RE.finditer(code):
            name = m.group(1)
            if path.name in ("mod.rs", "lib.rs", "main.rs"):
                base = path.parent
            else:
                base = path.with_suffix("")
            out.add(base / f"{name}.rs")
            out.add(base / name / "mod.rs")
    return out


def _count_unwrap_lines(*roots: pathlib.Path) -> int:
    """Lines matching `.unwrap()`/`.expect(` in non-test Rust under `roots`.

    Files under a `tests/` directory and files a `#[cfg(test)] mod name;`
    declaration pulls in are skipped whole; inside a file, each `#[cfg(test)]`
    item and each item under an `#[expect(clippy::unwrap_used | expect_used)]`
    is skipped and the scan RESUMES after it, because the marker decorates a
    trailing test module, a one-line `mod tests;`, and a statement inside a
    live function, and only the first of those three is the end of the
    production code in that file.
    """
    total = 0
    for root in roots:
        test_only = _test_only_files(root)
        for path in sorted(root.rglob("*.rs")):
            if "tests" in path.relative_to(root).parts or path in test_only:
                continue
            text = path.read_text(encoding="utf-8")
            raw = text.splitlines()
            code = _rust_code_lines(text)
            i = 0
            while i < len(raw):
                marker = _CFG_TEST_RE.search(code[i])
                if marker:
                    i = _end_of_cfg_test_item(code, i, marker.end())
                    continue
                sanctioned = _expect_panic_marker(code, i)
                if sanctioned is not None:
                    i = _end_of_cfg_test_item(code, *sanctioned)
                    continue
                if _UNWRAP_RE.search(raw[i]):
                    total += 1
                i += 1
    return total


def test_ffi_layer_has_no_unwrap_or_expect():
    """HARD RULE: zero in `bindings/python/src`, zero forever."""
    n = _count_unwrap_lines(_ROOT / "bindings" / "python" / "src")
    assert n == 0, (
        f"{n} unwrap()/expect( in bindings/python/src. A panic here costs the "
        "host a record or a connection. Return a PyResult error instead."
    )


def test_core_crates_have_no_unwrap_or_expect():
    """HARD RULE: zero in non-test `crates/*/src` (was a budget of 172)."""
    n = _count_unwrap_lines(*sorted((_ROOT / "crates").glob("*/src")))
    assert n == 0, (
        f"{n} unwrap()/expect( in non-test crates/*/src outside any "
        "`#[expect(clippy::expect_used, reason)]` item. Prove the branch by "
        "pattern (`let Some(x) = .. else`), or if it is truly unreachable, "
        "state the invariant in `expect(..)` under that attribute so the reason "
        "is reviewable. The compiler enforces the same rule through "
        "clippy::unwrap_used / expect_used; this scan keeps the snapshot honest."
    )


def test_a_wrapped_expect_attribute_still_sanctions_its_item(tmp_path: pathlib.Path):
    """The sanctioned shape is whatever rustfmt makes of it: a `reason` past the
    line width puts `clippy::expect_used` on its own line, and the item under
    it is still exempt. Judging only the attribute's first line counted the
    `.expect(` below and failed the hard rule while clippy itself passed."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "lib.rs").write_text(
        '#[expect(clippy::expect_used, reason = "length was just checked")]\n'
        "fn short(v: &[u8]) -> u8 {\n"
        '    *v.first().expect("length was just checked")\n'
        "}\n"
        "\n"
        "#[expect(\n"
        "    clippy::expect_used,\n"
        '    reason = "the slice was checked non-empty two lines above, so first() cannot fail"\n'
        ")]\n"
        "fn long(v: &[u8]) -> u8 {\n"
        '    *v.first().expect("checked non-empty above")\n'
        "}\n"
        "\n"
        '#[expect(clippy::string_slice, reason = "not a panic lint")]\n'
        "fn other(v: &[u8]) -> u8 {\n"
        '    *v.first().expect("this one is counted")\n'
        "}\n",
        encoding="utf-8",
    )
    assert _count_unwrap_lines(src) == 1


def test_cfg_test_mod_declaration_marks_the_file_test_only():
    """The counter must not count `semantic/tests.rs`: `mod tests;` pulls it in
    under `#[cfg(test)]`, and the file itself carries no marker."""
    root = _ROOT / "crates" / "wardex-protocol" / "src"
    assert root / "semantic" / "tests.rs" in _test_only_files(root)


# --------------------------------------------------------------------------
# Build and lint configuration — the rules in Cargo.toml hold for every crate
# --------------------------------------------------------------------------

try:
    import tomllib as _toml
except ImportError:  # Python 3.10; pytest itself depends on tomli there
    import tomli as _toml  # type: ignore[import-not-found,no-redef]  # the 3.10 floor has no tomllib

_ROOT_MANIFEST = _ROOT / "Cargo.toml"


def _member_manifests() -> list[pathlib.Path]:
    with _ROOT_MANIFEST.open("rb") as fh:
        members = _toml.load(fh)["workspace"]["members"]
    return [_ROOT / m / "Cargo.toml" for m in members]


def test_every_workspace_member_inherits_lints_and_publish():
    """HARD RULE: a member that forgets `[lints] workspace = true` silently
    opts out of every rule in the root Cargo.toml, and one that forgets
    `publish.workspace = true` could be uploaded to crates.io by accident."""
    missing = []
    for path in _member_manifests():
        with path.open("rb") as fh:
            manifest = _toml.load(fh)
        if manifest.get("lints", {}).get("workspace") is not True:
            missing.append(f"{path.relative_to(_ROOT)}: [lints] workspace = true")
        if manifest.get("package", {}).get("publish", {}) not in ({"workspace": True}, False):
            missing.append(f"{path.relative_to(_ROOT)}: publish.workspace = true")
    with _ROOT_MANIFEST.open("rb") as fh:
        root_publish = _toml.load(fh)["workspace"]["package"].get("publish")
    if root_publish is not False:
        missing.append("Cargo.toml: [workspace.package] publish = false")
    assert not missing, "\n".join(missing)


def test_abi3_floor_equals_the_declared_python_floor():
    """HARD RULE: the extension's stable-ABI floor (`abi3-py3NN` on pyo3) and
    `requires-python` are one fact stated in two files. A floor above
    `requires-python` ships a wheel pip refuses on the oldest supported Python."""
    with _ROOT_MANIFEST.open("rb") as fh:
        pyo3 = _toml.load(fh)["workspace"]["dependencies"]["pyo3"]
    abi3 = [f for f in pyo3["features"] if f.startswith("abi3-py")]
    assert len(abi3) == 1, abi3
    abi3_floor = abi3[0].removeprefix("abi3-py")
    with _PYPROJECT.open("rb") as fh:
        requires = _toml.load(fh)["project"]["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", requires)
    assert floor is not None, requires
    declared = floor.group(1) + floor.group(2)
    assert abi3_floor == declared, f"pyo3 {abi3[0]} vs requires-python {requires}"


def test_rust_crates_stay_exhaustive_and_undeprecated():
    """HARD RULE: these crates have no consumers outside this workspace, so a
    new variant or field must break every match and literal that has to decide
    what to do with it, and a changed signature is fixed at its callers in the
    same change instead of deprecated."""
    attr = re.compile(r"#\[\s*(?:cfg_attr\([^]]*?,\s*)?(?:non_exhaustive|deprecated)\b")
    hits = []
    for root in (_ROOT / "crates", _ROOT / "bindings"):
        for path in root.rglob("*.rs"):
            # Blanked code, so a mention in a comment or a string is not a hit.
            for i, line in enumerate(_rust_code_lines(path.read_text(encoding="utf-8")), 1):
                if attr.search(line):
                    hits.append(f"{path.relative_to(_ROOT)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_nothing_sets_panic_abort():
    """HARD RULE: the FFI shield and the fail-closed PII scrub catch panics;
    under `panic = "abort"` a caught panic is a dead host process. The gate is
    the `compile_error!` behind `cfg(not(panic = "unwind"))` in
    `bindings/python/src/lib.rs` and `crates/wardex-pipeline/src/lib.rs`, which
    stops any build that sets abort by whatever route (profile, RUSTFLAGS, a
    `CARGO_PROFILE_*_PANIC` variable). This scan is the early warning that
    names the file and line before a build has to fail."""
    pattern = re.compile(r"""panic\s*=\s*['"]abort['"]|panic=abort|CARGO_PROFILE_\w+_PANIC""")
    hits = []
    candidates = list(_ROOT.rglob("Cargo.toml")) + list(_ROOT.glob(".cargo/config*"))
    candidates += list((_ROOT / ".github").rglob("*.yml")) + list(
        (_ROOT / ".github").rglob("*.yaml")
    )
    candidates += list((_ROOT / "scripts").glob("*.sh")) + [_ROOT / "pyproject.toml", _PYPROJECT]
    for path in candidates:
        if "target" in path.parts or ".venv" in str(path) or not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if pattern.search(code):
                hits.append(f"{path.relative_to(_ROOT)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


# --------------------------------------------------------------------------
# P5 — module length
# --------------------------------------------------------------------------

_MODULE_CEILING = 800

#: Source modules over the ceiling on 2026-09-11 and their sizes (newline
#: count, as `wc -l`). Each may only shrink; a module not listed here may not
#: cross the ceiling; a module that drops below it is removed from this dict.
_OVERSIZED_BUDGET = {
    "_assembly/_units.py": 2097,
    "_adapters/_assembler.py": 1929,
    # Lowered from 1588 when the span pieces its later span kinds share with it
    # (`_open_child`, the start instant, the error sentences) moved out to
    # `_openai_agents_kinds.py` beside those kinds, with the model call's join.
    "_adapters/_openai_agents.py": 1465,
    "_client.py": 1565,
    "_adapters/_context.py": 1305,
    "_interceptors/_seam.py": 1153,
    "_adapters/_langgraph.py": 1035,
    # Raised by exactly one member (h2_request_evicted, 53): the closed
    # Limitation enum cannot move a member out, and the census demands each
    # one carry its own provenance docstring in this module. Every line of
    # prose around the member stayed on the diet. Lowered to 1083 when
    # provider_inferred (54) paid for its docstring by rewrapping two
    # sections' docstrings to the line length, words unchanged. Lowered to 1081
    # when frame_parse_failed gained its HTTP/1 emitter and a fifth session-table
    # emit site was added, each paid for the same way.
    "_assembly/_integrity.py": 1078,
    "_assembly/_builder.py": 963,
    "_adapters/_anthropic_agent_sdk.py": 924,
}


def _module_sizes() -> dict[str, int]:
    return {
        p.relative_to(_PY_SRC).as_posix(): p.read_text(encoding="utf-8").count("\n")
        for p in sorted(_PY_SRC.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def test_no_new_module_crosses_the_ceiling():
    """HARD RULE for everything not on the recorded list."""
    offenders = {
        m: n
        for m, n in _module_sizes().items()
        if n > _MODULE_CEILING and m not in _OVERSIZED_BUDGET
    }
    assert not offenders, (
        f"modules newly over {_MODULE_CEILING} lines: {offenders}. Split the "
        "module; the recorded list in this file is closed to additions."
    )


def test_recorded_oversized_modules_only_shrink():
    """BUDGET: each recorded module is at or below its recorded size."""
    sizes = _module_sizes()
    grew = {
        m: (sizes.get(m, 0), cap) for m, cap in _OVERSIZED_BUDGET.items() if sizes.get(m, 0) > cap
    }
    assert not grew, (
        f"recorded oversized modules grew (now, cap): {grew}. Move the new code "
        "out rather than in — these files are on a diet."
    )
    stale = {
        m: (sizes.get(m, 0), cap) for m, cap in _OVERSIZED_BUDGET.items() if sizes.get(m, 0) < cap
    }
    if stale:
        pytest.fail(
            f"recorded modules shrank (now, cap): {stale}. Lower their entries in "
            "_OVERSIZED_BUDGET to the new sizes in this commit (delete the entry "
            f"if now ≤ {_MODULE_CEILING}) and say so in the commit body."
        )


# --------------------------------------------------------------------------
# P2/P3 — hand-run checks exist only if AGENTS.md says so
# --------------------------------------------------------------------------


#: A Rust `#[ignore]` attribute, anchored at the start of its line. Anchoring
#: is load-bearing: `crates/wardex-protocol/src/semantic/mod.rs` quotes
#: `#[ignore]` in the prose of a `//!` doc comment, and an unanchored substring
#: scan walks forward from that line into the next private helper and demands
#: `AGENTS.md` list a function that is not a check at all.
_RUST_IGNORE_RE = re.compile(r"^\s*#\[\s*ignore\b")
_RUST_COMMENT_RE = re.compile(r"^\s*(?://|/\*|\*)")
_RUST_FN_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r'(?:extern\s+"[^"]*"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)'
)


def _ignored_rust_tests() -> tuple[list[str], list[str]]:
    """`#[ignore]`d Rust tests under `crates/` and `bindings/`.

    Returns the enclosing function names and, separately, any `#[ignore]` whose
    function this scanner could not name. The second list must stay empty: an
    unresolved site would drop out of the check silently, which is the one way
    a hand-run probe can go missing without anything turning red.

    `tests/` directories are walked, unlike `_count_unwrap_lines` — one of the
    probes is an integration test at `crates/wardex-pipeline/tests/pii_perf.rs`
    and skipping its directory would hide it.
    """
    names: list[str] = []
    unresolved: list[str] = []
    for root in (_ROOT / "crates", _ROOT / "bindings"):
        for path in sorted(root.rglob("*.rs")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if _RUST_COMMENT_RE.match(line) or not _RUST_IGNORE_RE.match(line):
                    continue
                for nxt in lines[i + 1 : i + 25]:
                    if _RUST_COMMENT_RE.match(nxt) or nxt.strip().startswith("#["):
                        continue
                    found = _RUST_FN_RE.match(nxt)
                    if found:
                        names.append(found.group(1))
                    elif nxt.strip():
                        unresolved.append(f"{path.relative_to(_ROOT).as_posix()}:{i + 1}")
                    break
    return names, unresolved


def test_every_hand_run_check_is_listed_in_agents_md():
    """HARD RULE: each `e2e_*.py` / `bench_*.py` file name and each `#[ignore]`d
    Rust test name appears in AGENTS.md."""
    agents = _AGENTS.read_text(encoding="utf-8")
    rust_names, unresolved = _ignored_rust_tests()
    assert not unresolved, (
        f"#[ignore] attributes whose test this scanner could not name: {unresolved}. "
        "Until it can, those probes are exempt from the AGENTS.md check without "
        "anyone being told. Teach _RUST_FN_RE the new spelling."
    )
    candidates = {p.name for pattern in ("e2e_*.py", "bench_*.py") for p in _TESTS.glob(pattern)}
    candidates.update(rust_names)
    unlisted = sorted(name for name in candidates if name not in agents)
    assert not unlisted, (
        f"hand-run checks not listed in AGENTS.md 'Checks no gate runs': {unlisted}. "
        "Add each with the condition that triggers running it. Unlisted, it does "
        "not exist. Rust entries are `#[ignore]`d tests: no gate passes "
        "`--ignored`, so nothing but AGENTS.md will ever tell anyone they exist."
    )


# --------------------------------------------------------------------------
# P2/P5 — the wheel has no runtime Python dependencies
# --------------------------------------------------------------------------


def test_wheel_declares_zero_runtime_dependencies():
    """HARD RULE: no `dependencies = [...]` under `[project]`.

    Text, not tomllib: the floor is 3.10 and tomllib is 3.11+. The `[project]`
    table runs until the next `[` header; a `dependencies` key inside it is the
    only spelling PEP 621 accepts, so a line match is exact.
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\n(.*?)(?=^\[)", text, re.S | re.M)
    assert project, "pyproject.toml has no [project] table"
    assert not re.search(r"^\s*dependencies\s*=", project.group(1), re.M), (
        "sdks/python/pyproject.toml declares runtime dependencies. The wheel "
        "ships with zero: a host's resolver must never have to negotiate with "
        "ours. Vendor, move it to Rust, or make it optional."
    )


# --------------------------------------------------------------------------
# P4 — public names are documented
# --------------------------------------------------------------------------

#: Public names without a docstring on 2026-09-11. Down only: writing the
#: docstring removes the name from this set in the same commit.
_UNDOCUMENTED_BUDGET = frozenset(
    {
        "get_traceparent",
        "get_trace_headers",
        "WardexAsgiMiddleware",
        "WardexWsgiMiddleware",
    }
)


def test_public_docstring_debt_only_falls():
    """BUDGET: the set of undocumented `__all__` names is exactly the recorded one
    or smaller. Imports the package (docstrings are runtime attributes); the
    ratchet is over Python source only, so a stale wheel cannot move it."""
    import inspect

    import wardex_sdk

    undocumented = {
        name
        for name in wardex_sdk.__all__
        if not (inspect.getdoc(getattr(wardex_sdk, name)) or "").strip()
    }
    new = undocumented - _UNDOCUMENTED_BUDGET
    assert not new, (
        f"public names added without a docstring: {sorted(new)}. help() is the "
        "first documentation a host reads."
    )
    paid = _UNDOCUMENTED_BUDGET - undocumented
    if paid:
        pytest.fail(
            f"docstrings landed for {sorted(paid)}. Remove them from "
            "_UNDOCUMENTED_BUDGET in this commit and say so in the commit body."
        )


# --------------------------------------------------------------------------
# P4 — a skip names its reason on the marker that carries it
# --------------------------------------------------------------------------

#: Marker calls that stop a test from running.
_SKIP_MARKERS = frozenset(
    {
        "pytest.mark.skip",
        "pytest.mark.skipif",
        "pytest.mark.xfail",
        "pytest.skip",
        "pytest.xfail",
        "pytest.importorskip",
    }
)

#: Markers whose reason may be the first positional argument: `pytest.skip` and
#: `pytest.xfail` take their message there, not as `reason=`.
_POSITIONAL_REASON = frozenset({"pytest.skip", "pytest.xfail"})

#: Markers that are legal without a call — a bare `@pytest.mark.skip` is a site
#: with no reason anywhere, so it is caught rather than skipped over.
_BARE_MARKERS = frozenset({"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail"})

#: Skip marker sites under `sdks/python/tests` on 2026-09-13. Down only. A
#: skipped test is a test that does not run, so adding one is an edit to the
#: constitution, not a detail of the change that wanted it. This also keeps the
#: scan honest: a scanner that stopped seeing markers would report 0 and fail
#: here rather than pass an empty check.
_SKIP_SITE_BUDGET = 10


def _dotted_name(node: ast.AST) -> str | None:
    """`pytest.mark.skipif` for that attribute chain, None for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _call_has_reason(call: ast.Call, marker: str) -> bool:
    """True if the call carries a non-empty reason on the marker itself."""
    values = [kw.value for kw in call.keywords if kw.arg == "reason"]
    if not values and marker in _POSITIONAL_REASON and call.args:
        values = [call.args[0]]
    for value in values:
        if not isinstance(value, ast.Constant):
            return True  # an f-string or a computed reason: present, unreadable
        if isinstance(value.value, str) and value.value.strip():
            return True
    return False


def _skip_sites() -> list[tuple[str, int, str, bool]]:
    """Every skip marker site under the Python test tree.

    Yields `(file, line, marker, has_reason)`. An alias — `fork_only =
    pytest.mark.skipif(..., reason=...)` applied as `@fork_only` at thirteen
    further lines — is one site, at its definition, because that is the one
    place the reason is written and the one place it can rot.
    """
    sites: list[tuple[str, int, str, bool]] = []
    for path in sorted(_TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                marker = _dotted_name(node.func)
                if marker in _SKIP_MARKERS:
                    sites.append((path.name, node.lineno, marker, _call_has_reason(node, marker)))
            elif isinstance(node, ast.Attribute) and id(node) not in called:
                marker = _dotted_name(node)
                if marker in _BARE_MARKERS:
                    sites.append((path.name, node.lineno, marker, False))
    return sites


def test_every_skip_marker_names_its_reason():
    """HARD RULE: no skip, xfail or importorskip without a reason on its marker."""
    reasonless = sorted(
        f"{marker} at {file}:{line}" for file, line, marker, ok in _skip_sites() if not ok
    )
    assert not reasonless, (
        f"skip markers carrying no reason: {reasonless}. Write the reason on the "
        "marker itself — `reason=...`, or the first argument of pytest.skip. A "
        "reason kept somewhere else is a reason nobody reads when deciding "
        "whether the test can run again."
    )


def test_skip_marker_site_count_only_falls():
    """BUDGET: skip marker sites under `sdks/python/tests`, down only."""
    n = len(_skip_sites())
    assert n <= _SKIP_SITE_BUDGET, (
        f"skip markers rose to {n} sites (budget {_SKIP_SITE_BUDGET}). A skip is "
        "coverage the suite reports as green and never ran. Make the test work "
        "on the platform, or raise the budget in a deliberate `quality:` commit."
    )
    if n < _SKIP_SITE_BUDGET:
        pytest.fail(
            f"skip markers fell to {n} sites. Lower _SKIP_SITE_BUDGET to {n} in "
            "this commit so the ratchet holds the new position — and say so in "
            "the commit body."
        )
