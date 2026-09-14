#!/usr/bin/env bash
# Print the quality budgets of the current tree, as JSON. README.md describes
# what each one holds; sdks/python/tests/test_quality_ratchets.py enforces the
# ones that may move in one direction only.
#
# Read-only and dependency-free on purpose: grep, awk and the system python3
# only -- no venv, no cargo, and nothing imported from the built package, so a
# stale wheel cannot change a number here. It does read exactly one build
# product, and only its size: `native_module` stats `_wardex_native.abi3.so`
# when a local build has left one in the source tree, and reports
# `present: false` with a null size when it has not. The weekly radar diffs
# this output against the recorded values; nothing in this script enforces
# anything. A budget becomes a real test when the measurement is stable enough
# to fail a build on.
#
# `--measure` adds the numbers that cannot be read off the source: the import
# wall clock, the test suite wall clock, and the crate dependency split. Those
# need the built venv, the native module, or cargo with a warm registry, which
# is exactly why they are not on the default path.
set -euo pipefail

MEASURE=0
for arg in "$@"; do
  case "$arg" in
    --measure) MEASURE=1 ;;
    *) echo "usage: $(basename "$0") [--measure]" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_SRC="sdks/python/src/wardex_sdk"
PY_TESTS="sdks/python/tests"
PY_PROJECT="sdks/python/pyproject.toml"
NATIVE_SO="$PY_SRC/_wardex_native.abi3.so"

# Is a file part of the tree an outside host can clone? Several inputs below
# are gitignored build products, and a number derived from a file nobody else
# can see is not verifiable from outside, so the row says so rather than
# quietly printing the number.
tracked() {
  if git ls-files --error-unmatch "$1" >/dev/null 2>&1; then echo true; else echo false; fi
}

# --- P1: panic surface -------------------------------------------------------
# Non-test Rust only: files under a `tests/` dir are skipped whole, and inside a
# file each `#[cfg(test)]` ITEM is skipped, after which the scan resumes.
#
# It resumes because `#[cfg(test)]` does not always open a trailing test module.
# In `crates/wardex-protocol/src/semantic/mod.rs` it decorates the one-line
# declaration `mod tests;` on line 13 of 532, and in
# `crates/wardex-pipeline/src/pii/walk.rs` it decorates a test-only `if` in the
# middle of a live function. The awk one-liner this replaced exited the file at
# the first marker and so reported 0 for that 532-line module instead of 3 --
# three `.unwrap()` calls on the SSE dispatch the FFI reaches. An
# under-reporting counter looks exactly like a ratchet holding.
#
# This is character-for-character the method in
# `sdks/python/tests/test_quality_ratchets.py` (`_count_unwrap_lines` and the
# two helpers it calls). The test enforces, this reports; if they ever disagree
# the radar is diffing against a number no gate defends, so keep them identical
# and run both after touching either.
count_unwrap() {
  python3 - "$@" <<'PY'
import os
import re
import sys

UNWRAP_RE = re.compile(r"\.unwrap\(\)|\.expect\(")
CFG_TEST_RE = re.compile(r"#\[\s*cfg\(test\)\s*\]")
RAW_STR_RE = re.compile(r'b?r(#*)"')
CHAR_LIT_RE = re.compile(r"'(?:\\.|[^\\'])'")
IDENT_CHAR_RE = re.compile(r"[A-Za-z0-9_]")


def blanked(chunk):
    return "".join("\n" if c == "\n" else " " for c in chunk)


def code_lines(text):
    """`text` line for line with comments and literals blanked out, so that a
    `{` inside a JSON fixture or a doc comment cannot be mistaken for a block."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and text[i + 1 : i + 2] == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
            out.append(blanked(text[i:end]))
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
            out.append(blanked(text[i:end]))
            i = end
            continue
        raw = RAW_STR_RE.match(text, i)
        if raw and not (i and IDENT_CHAR_RE.match(text[i - 1])):
            close = '"' + raw.group(1)
            end = text.find(close, raw.end())
            end = n if end < 0 else end + len(close)
            out.append(blanked(text[i:end]))
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
            out.append(blanked(text[i:end]))
            i = end
            continue
        if ch == "'":
            lit = CHAR_LIT_RE.match(text, i)
            if lit:  # a char literal; a lifetime has no closing quote
                out.append(blanked(lit.group(0)))
                i = lit.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out).splitlines()


def end_of_cfg_test_item(code, start, col):
    """Index of the first line after the item a `#[cfg(test)]` decorates: where
    its braces balance, or its semicolon for a declaration that has no block."""
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


total = 0
for root in sys.argv[1:]:
    for base, dirs, files in os.walk(root):
        rel = os.path.relpath(base, root).split(os.sep)
        if "tests" in rel:
            continue
        for name in sorted(files):
            if not name.endswith(".rs"):
                continue
            with open(os.path.join(base, name), encoding="utf-8") as fh:
                text = fh.read()
            raw = text.splitlines()
            code = code_lines(text)
            i = 0
            while i < len(raw):
                marker = CFG_TEST_RE.search(code[i])
                if marker:
                    i = end_of_cfg_test_item(code, i, marker.end())
                    continue
                if UNWRAP_RE.search(raw[i]):
                    total += 1
                i += 1
print(total)
PY
}
unwrap_crates=$(count_unwrap crates/*/src)
unwrap_bindings=$(count_unwrap bindings/python/src)

# --- P5: module length -------------------------------------------------------
module_sizes=$(find "$PY_SRC" -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 wc -l | grep -v ' total$' | sort -rn)
over_800=$(echo "$module_sizes" | awk '$1 > 800' | wc -l | tr -d ' ')
largest_line=$(echo "$module_sizes" | head -1)
largest_lines=$(echo "$largest_line" | awk '{print $1}')
largest_file=$(echo "$largest_line" | awk '{print $2}' | sed "s#^$PY_SRC/##")
top_modules=$(echo "$module_sizes" | awk '$1 > 800' \
  | sed "s#$PY_SRC/##" | awk '{printf "%s{\"file\":\"%s\",\"lines\":%s}", (NR>1?",":""), $2, $1}')

# --- P5: layering debt --------------------------------------------------------
# The §3 layering-debt row, read off `_LAYER_RANK_DEBT` in test_import_graph.py
# the same AST way as the budgets above. Both numbers are emitted because they
# are not the same number, and conflating them is the trap that record sets:
# five NAMED EDGES point the wrong way and they cost six OCCURRENCES, because
# `from .._assembly import PatchSet` resolves to both the package and a symbol
# re-exported from it and is deliberately counted twice. A reader who diffs an
# edge count against an occurrence total sees a drift that is not there.
layering_debt=$(python3 - "$PY_TESTS/test_import_graph.py" <<'PY'
import ast, json, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
out = {"edges": None, "occurrences": None}
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if not any(isinstance(t, ast.Name) and t.id == "_LAYER_RANK_DEBT" for t in node.targets):
        continue
    if isinstance(node.value, ast.Dict):
        out = {
            "edges": len(node.value.keys),
            "occurrences": sum(
                v.value
                for v in node.value.values
                if isinstance(v, ast.Constant) and isinstance(v.value, int)
            ),
        }
print(json.dumps(out))
PY
)

# --- P5: import-graph budgets -------------------------------------------------
budget_totals=$(python3 - "$PY_TESTS/test_import_graph.py" <<'PY'
import ast, sys, json
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
out = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if not (isinstance(t, ast.Name) and t.id.endswith("_BUDGET")):
                continue
            if isinstance(node.value, ast.Dict):
                total = 0
                for v in node.value.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, int):
                        total += v.value
                out[t.id] = {"files": len(node.value.keys), "total": total}
print(json.dumps(out))
PY
)

# --- P5, P6: Rust crate count -------------------------------------------------
# Cargo.lock is gitignored, so `tracked` is false and a host cloning this repo
# cannot reproduce `packages` at all without resolving a lockfile of their own,
# which may resolve differently. The splitting of those packages into link-time
# and build-only needs `cargo metadata`, which needs cargo and a warm registry
# cache, so it lives behind --measure and is null here.
if [ -f Cargo.lock ]; then
  lock_packages=$(grep -c '^\[\[package\]\]' Cargo.lock | tr -d ' ')
else
  lock_packages=null
fi
lock_tracked=$(tracked Cargo.lock)

# --- P2, P5: zero runtime dependencies ----------------------------------------
# The wheel declares no runtime dependencies. Read as the absence of a
# `dependencies` key in the `[project]` table rather than as a count, because
# the honest statement is "the key is not there", and an added empty list
# would be a change worth seeing.
runtime_deps=$(python3 - "$PY_PROJECT" <<'PY'
import json, re, sys

lines, in_project = [], False
for raw in open(sys.argv[1], encoding="utf-8").read().splitlines():
    line = raw.strip()
    if line.startswith("["):
        in_project = line == "[project]"
        continue
    if in_project:
        lines.append(line)

body, depth = [], 0
for line in lines:
    if depth == 0 and not re.match(r"dependencies\s*=", line):
        continue
    body.append(line)
    depth += line.count("[") - line.count("]")
    if depth <= 0 and body:
        break
count = sum(1 for line in body if line.startswith(('"', "'")))
print(json.dumps({"declared_in_project_table": bool(body), "count": count}))
PY
)

# --- P4: public surface -------------------------------------------------------
all_len=$(python3 - "$PY_SRC/__init__.py" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if any(getattr(t, "id", "") == "__all__" for t in node.targets):
        print(len(node.value.elts))
        break
else:
    print(-1)
PY
)
examples=$(find examples -maxdepth 1 -name '*.py' | wc -l | tr -d ' ')

# --- P4: public names without a docstring -------------------------------------
# By AST over the source, never by importing: importing would need the built
# native module, and then a stale wheel could move the number. The rule below
# reproduces what `inspect.getdoc` reports at runtime, which is what the
# enforcing test in test_quality_ratchets.py sees -- a class counts as
# documented when it has its own docstring, when `@dataclass` synthesises one
# for it, or when it inherits one from a base. A base this package does not
# define (Enum, ABC, Protocol) is assumed to carry a docstring, because every
# stdlib base does; that assumption is the one place this can disagree with a
# live `inspect.getdoc`.
undocumented=$(python3 - "$PY_SRC" <<'PY'
import ast, json, os, sys

PKG = sys.argv[1]
_cache = {}


def _tree(mod):
    if mod in _cache:
        return _cache[mod]
    base = os.path.join(PKG, *[p for p in mod.split(".") if p])
    path = base + ".py" if os.path.isfile(base + ".py") else os.path.join(base, "__init__.py")
    tree = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    _cache[mod] = (tree, os.path.basename(path) == "__init__.py")
    return _cache[mod]


def _target(mod, node):
    """Absolute module name (relative to the package root) a relative import names."""
    _, is_pkg = _tree(mod)
    parts = [p for p in mod.split(".") if p]
    if not is_pkg:
        parts = parts[:-1]
    for _ in range(node.level - 1):
        parts = parts[:-1]
    return ".".join(parts + ([node.module] if node.module else []))


def _is_dataclass(node):
    for dec in node.decorator_list:
        f = dec.func if isinstance(dec, ast.Call) else dec
        if getattr(f, "id", getattr(f, "attr", "")) == "dataclass":
            return True
    return False


def documented(mod, name, depth=0):
    if depth > 8:
        return True
    tree, _ = _tree(mod)
    if tree is None:
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_docstring(node) is not None
        if isinstance(node, ast.ClassDef) and node.name == name:
            if ast.get_docstring(node) is not None or _is_dataclass(node):
                return True
            for b in node.bases:
                if not isinstance(b, ast.Name) or documented(mod, b.id, depth + 1):
                    return True
            return False
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if (a.asname or a.name) == name:
                    if node.level == 0:
                        return True
                    return documented(_target(mod, node), a.name, depth + 1)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == name:
                    return True
    return True


root, _ = _tree("")
names = []
for node in ast.walk(root):
    if not isinstance(node, ast.Assign):
        continue
    if any(getattr(t, "id", "") == "__all__" for t in node.targets):
        names = [e.value for e in node.value.elts]
        break
bad = sorted(n for n in names if not documented("", n))
print(json.dumps({"count": len(bad), "names": bad}))
PY
)

# --- P2: native module size ---------------------------------------------------
# Gitignored and locally built, so it is absent on a fresh clone and the bytes
# differ per host and per build profile. Emit null instead of failing, and say
# it is untracked so nobody reads the number as reproducible.
if [ -f "$NATIVE_SO" ]; then
  so_bytes=$(stat -f %z "$NATIVE_SO" 2>/dev/null || stat -c %s "$NATIVE_SO")
  native_module="{\"present\": true, \"bytes\": $so_bytes, \"tracked\": $(tracked "$NATIVE_SO")}"
else
  native_module='{"present": false, "bytes": null, "tracked": false}'
fi

# --- dated gaps: files that do not exist yet ----------------------------------
# Two rows of §3 are promised artefacts, not yet written. The radar watches
# these flags so the day either file appears its row stops reading "unmeasured"
# without anyone having to remember to look.
exists() { if [ -e "$1" ]; then echo true; else echo false; fi; }
perf_baseline=$(exists quality/perf-baseline.json)
fault_matrix=$(exists "$PY_TESTS/test_fault_matrix.py")

# --- P2, P3: tests that do not run --------------------------------------------
# A skipped test and an `#[ignore]`d probe both look green while asserting
# nothing, so both counts belong beside the hand-run list.
#
# The skip scan is by AST, not by grep, for one concrete reason: the enforcing
# test spells these marker names in its own source, so a grep counts the
# counter and reports six sites that do not exist. An alias assigned once and
# applied many times is one site, at its assignment, matching the rule in
# test_quality_ratchets.py so the two numbers can be compared.
skip_sites=$(python3 - "$PY_TESTS" <<'PY'
import ast, os, sys

MARKERS = {
    "pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail",
    "pytest.skip", "pytest.xfail", "pytest.importorskip",
}
BARE = {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail"}


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


count = 0
for base, dirs, files in os.walk(sys.argv[1]):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(base, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                count += dotted(node.func) in MARKERS
            elif isinstance(node, ast.Attribute) and id(node) not in called:
                count += dotted(node) in BARE
print(count)
PY
)

# --- P2/P3: hand-run checks must be listed in AGENTS.md -----------------------
# Two kinds of check run only when a person runs them: the uncollected
# `e2e_*.py` / `bench_*.py` files, and `#[ignore]`d Rust probes, which no gate
# reaches because no gate passes `--ignored`. Both are counted here. The
# `#[ignore]` scan is anchored at the start of a line, because one module
# quotes `#[ignore]` in the prose of a `//!` doc comment, and an unanchored
# scan walks from there into the next unrelated function.
hand_run=$(python3 - "$PY_TESTS" <<'PY'
import json, os, re, sys

IGNORE_RE = re.compile(r"^\s*#\[\s*ignore\b")
COMMENT_RE = re.compile(r"^\s*(?://|/\*|\*)")
FN_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r'(?:extern\s+"[^"]*"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)'
)

names, unresolved, probes = [], [], 0
for root in ("crates", "bindings"):
    for base, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".rs"):
                continue
            path = os.path.join(base, fname)
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            for i, line in enumerate(lines):
                if COMMENT_RE.match(line) or not IGNORE_RE.match(line):
                    continue
                probes += 1
                for nxt in lines[i + 1 : i + 25]:
                    if COMMENT_RE.match(nxt) or nxt.strip().startswith("#["):
                        continue
                    found = FN_RE.match(nxt)
                    if found:
                        names.append(found.group(1))
                    elif nxt.strip():
                        unresolved.append(f"{path}:{i + 1}")
                    break

tests = sys.argv[1]
candidates = {
    f for f in os.listdir(tests)
    if f.endswith(".py") and (f.startswith("e2e_") or f.startswith("bench_"))
}
candidates.update(names)
agents = open("AGENTS.md", encoding="utf-8").read()
print(json.dumps({
    "unlisted_in_agents_md": sorted(c for c in candidates if c not in agents),
    "rust_ignored_probes": probes,
    "rust_ignored_names_unresolved": sorted(unresolved),
}))
PY
)

# --- informational ------------------------------------------------------------
# Nothing below has a row in §3. They are here as context for the radar, not as
# ratchets; anything that earns a ratchet moves up into its promise group.
py_tests=$(grep -h '^\s*\(async \)\?def test_' "$PY_TESTS"/test_*.py | wc -l | tr -d ' ')
rust_tests=$(grep -rh '#\[test\]' crates bindings --include='*.rs' | wc -l | tr -d ' ')
py_src_loc=$(find "$PY_SRC" -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 cat | wc -l | tr -d ' ')
py_test_loc=$(find "$PY_TESTS" -name '*.py' -not -path '*/__pycache__/*' -print0 \
  | xargs -0 cat | wc -l | tr -d ' ')
rust_loc=$(find crates bindings -name '*.rs' -print0 | xargs -0 cat | wc -l | tr -d ' ')
readme_lines=$(wc -l < README.md | tr -d ' ')
pin_re='^[[:space:]]*"(langgraph|openai-agents|claude-agent-sdk|langchain|langchain-core)[><=]'
pins=$(grep -E "$pin_re" "$PY_PROJECT" \
  | sed -E 's/^[[:space:]]*"([^"]+)".*/\1/' | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')

# --- --measure: the numbers that need a built tree ----------------------------
measured=null
if [ "$MEASURE" = 1 ]; then
  # Crate split. `cargo metadata` resolves with the workspace's default feature
  # set, so a different feature selection would drop optional dependencies this
  # does not see. Host-filtered and all-target counts both, because they differ
  # a lot: the wasm and protoc build blobs vanish under the host filter.
  # No cargo on this machine means no split, not a dead run: the import and
  # suite numbers below do not need it.
  if command -v cargo >/dev/null && command -v rustc >/dev/null; then
    host=$(rustc -vV | awk '/host:/{print $2}')
  else
    host=unknown
  fi
  split_py='
import json, sys
m = json.load(sys.stdin)
W = set(m["workspace_members"])
N = {n["id"]: n for n in m["resolve"]["nodes"]}
P = {p["id"]: p for p in m["packages"]}


def walk(kinds):
    seen, stack = set(), list(W)
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        for d in N[c]["deps"]:
            if kinds & {k["kind"] for k in d["dep_kinds"]}:
                stack.append(d["pkg"])
    return seen


link = walk({None}) - W
build_only = walk({None, "build", "dev"}) - walk({None, "dev"})
macros = {i for i in link if any("proc-macro" in t["kind"] for t in P[i]["targets"])}
print(json.dumps({
    "packages": len(m["packages"]),
    "workspace_members": len(W),
    "link_time": len(link),
    "link_time_excluding_proc_macros": len(link - macros),
    "build_only": len(build_only),
}))
'
  if [ "$host" = unknown ]; then
    host_split=null
    all_split=null
  else
    host_split=$(cargo metadata --format-version 1 --offline --filter-platform "$host" \
      | python3 -c "$split_py")
    all_split=$(cargo metadata --format-version 1 --offline | python3 -c "$split_py")
  fi

  # Import wall clock: three runs, median, in milliseconds. Warm, not cold --
  # nothing here drops the OS page cache, so do not report it as cold.
  import_samples=""
  for _ in 1 2 3; do
    us=$(uv run python -X importtime -c "import wardex_sdk" 2>&1 \
      | awk -F'|' '{gsub(/ /, "", $3)}
                   $3 == "wardex_sdk" {gsub(/ /, "", $2); v = $2}
                   END {print v + 0}')
    import_samples="$import_samples${import_samples:+,}$us"
  done
  import_stats=$(python3 -c "
import json, sys
s = sorted(int(v) for v in sys.argv[1].split(','))
print(json.dumps({'median_ms': round(s[len(s)//2]/1000.0, 1),
                  'samples_ms': [round(v/1000.0, 1) for v in s]}))" "$import_samples")

  # Test suite wall clock. Whole suite, default interpreter, no parallelism.
  suite_start=$(date +%s)
  suite_log=$(mktemp)
  if uv run pytest "$PY_TESTS" -q -p no:cacheprovider >"$suite_log" 2>&1; then
    suite_ok=true
  else
    suite_ok=false
  fi
  suite_s=$(( $(date +%s) - suite_start ))
  suite_summary=$(grep -Eo '[0-9]+ (passed|failed)[^"]*' "$suite_log" | tail -1 \
    | tr -d '"\\' || true)
  rm -f "$suite_log"

  measured=$(cat <<MEASURED
{
    "host_triple": "$host",
    "crate_split_host": $host_split,
    "crate_split_all_targets": $all_split,
    "import_wall_clock": $import_stats,
    "test_suite": {"wall_clock_s": $suite_s, "passed": $suite_ok, "summary": "$suite_summary"}
  }
MEASURED
)
fi

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
date=$(date +%F)

cat <<JSON
{
  "recorded": {"date": "$date", "commit": "$commit"},
  "p1_harmless": {
    "rust_unwrap_expect_nontest_crates": $unwrap_crates,
    "rust_unwrap_expect_bindings": $unwrap_bindings
  },
  "p5_changeable": {
    "py_modules_over_800": $over_800,
    "largest_module": {"file": "$largest_file", "lines": $largest_lines},
    "modules_over_800": [$top_modules],
    "import_graph_budgets": $budget_totals,
    "layering_debt": $layering_debt,
    "cargo_lock": {"packages": $lock_packages, "tracked": $lock_tracked},
    "runtime_dependencies": $runtime_deps
  },
  "p4_understandable": {
    "root_all_len": $all_len,
    "examples": $examples,
    "undocumented_public_names": $undocumented
  },
  "p2_p3_hand_run": {
    "hand_run_checks": $hand_run,
    "python_skip_sites": $skip_sites
  },
  "p2_efficient": {
    "native_module": $native_module
  },
  "dated_gaps": {
    "perf_baseline_json": $perf_baseline,
    "fault_matrix_test": $fault_matrix
  },
  "informational": {
    "py_tests": $py_tests,
    "rust_tests": $rust_tests,
    "py_src_loc": $py_src_loc,
    "py_test_loc": $py_test_loc,
    "rust_loc": $rust_loc,
    "readme_lines": $readme_lines,
    "adapter_pins": [$pins]
  },
  "measured": $measured
}
JSON
