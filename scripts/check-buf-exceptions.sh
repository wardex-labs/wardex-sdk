#!/usr/bin/env bash
# Fail when `buf.yaml` carries a `buf breaking` exception that excuses nothing.
#
# Why this exists: a deliberate wire break lands together with an `ignore` /
# `ignore_only` / `except` entry under `breaking:` in `buf.yaml`. Such an entry
# excuses a whole file (or, for `except`, a whole rule), not one field, and once the baseline contains the break there is
# nothing left to excuse — so an entry that outlives its change only switches
# the check off for every later break in that file. That happened: an entry
# stayed behind for three days while deleting any `envelope.proto` field
# passed CI, under a comment saying the opposite.
#
# The test: strip the exceptions and run the same comparison. If it still
# passes, the exceptions are stale. In the change that introduces a break the
# stripped run fails (the exception is doing its job) and this check is quiet;
# from the next change on, the baseline contains the break and it turns red
# until the entry is removed.
#
# Known limit: exceptions are stripped all at once. With several entries of
# which only some are stale, the stripped run still fails on the live one and
# the stale ones go unnoticed until it is removed too.
#
# Usage: scripts/check-buf-exceptions.sh <against>
#   <against> is what `buf breaking --against` takes, e.g.
#   '.git#format=git,ref=origin/main'
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <against>   (e.g. '.git#format=git,ref=origin/main')" >&2
  exit 2
fi
AGAINST="$1"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A directory, so the file inside can be named `*.yaml`: `buf --config` reads
# its argument as a path only when it ends in a YAML/JSON extension and as
# inline config data otherwise (measured: an extensionless temp path fails with
# "decode config file").
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/buf-stripped.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
STRIPPED="$SCRATCH/buf.yaml"

# Drop the exception keys (`ignore`, `ignore_only`, `except`) and everything
# nested under them, but only directly under the top-level `breaking:` section.
# `lint:` blocks carry keys of the same names and are none of this check's
# business, at the top level or under a module.
#
# No YAML parser on purpose: the CI job that runs this has none. So the walk
# reads exactly one shape — a plain block-style key under top-level `breaking:`
# — and the same pass counts every exception-looking key anywhere outside a
# `lint:` block, whatever its spelling (quoted, flow-style, explicit `? key`,
# a module-level `breaking:` override). If it sees a key it did not strip, or
# any flow-style YAML inside `breaking:`, it refuses instead of guessing: an
# exception this script cannot strip must never read as "no exceptions".
#
# Exit: 0 = stripped at least one, 3 = none present, 4 = a shape it cannot read.
set +e
awk '
  function indent(s) { match(s, /^ */); return RLENGTH }
  {
    code = $0
    sub(/\r$/, "", code)
    sub(/[ \t]*#.*/, "", code)          # the line without its comment
    blank = (code ~ /^[ \t]*$/)
  }
  !blank && code ~ /^[^ \t]/ {
    section = code; sub(/:.*/, "", section); skipping = 0
  }
  !blank {
    if (in_lint && indent(code) <= lint_indent) in_lint = 0
    if (code ~ /^ *lint *:/) { in_lint = 1; lint_indent = indent(code) }
  }
  {
    if (skipping) {
      if (blank || indent(code) > skip_indent) next
      # Block sequences may sit at the same indent as their key.
      if (indent(code) == skip_indent && code ~ /^ *- /) next
      skipping = 0
    }
    if (!blank && !in_lint) {
      if (section == "breaking" && code ~ /[{}\[\]]/) unreadable = 1
      if (code ~ /(^|[^A-Za-z_])(ignore|ignore_only|except)["\047]? *:/ ||
          code ~ /\? *["\047]?(ignore|ignore_only|except)([^A-Za-z_]|$)/) seen++
      if (section == "breaking" && code ~ /^ +(ignore|ignore_only|except) *:/) {
        stripped++; skipping = 1; skip_indent = indent(code); next
      }
    }
    print
  }
  END {
    if (unreadable || seen != stripped) exit 4
    exit stripped ? 0 : 3
  }
' buf.yaml > "$STRIPPED"
AWK_STATUS=$?
set -e

case "$AWK_STATUS" in
  0) ;;
  3)
    echo "buf.yaml has no breaking exceptions; nothing to go stale."
    exit 0
    ;;
  4)
    echo "buf.yaml has an \`ignore\` / \`ignore_only\` / \`except\` key, or flow-style" >&2
    echo "YAML inside \`breaking:\`, that this check cannot place. It reads only plain" >&2
    echo "block-style keys directly under the top-level \`breaking:\` section (and" >&2
    echo "skips \`lint:\` blocks). Write the exception in that shape so it can be" >&2
    echo "checked; an exception this check cannot strip is not allowed to pass." >&2
    exit 1
    ;;
  *)
    echo "could not read buf.yaml (awk exit $AWK_STATUS)" >&2
    exit 1
    ;;
esac

echo "buf.yaml has breaking exceptions; running the comparison without them:"
set +e
OUTPUT="$(buf breaking --against "$AGAINST" --config "$STRIPPED" 2>&1)"
STATUS=$?
set -e

case "$STATUS" in
  0)
    echo "STALE: \`buf breaking\` passes against the baseline with every exception" >&2
    echo "removed, so the exceptions in buf.yaml excuse nothing. Left in place they" >&2
    echo "silence the check for every later break in the files they name." >&2
    echo "Remove the \`ignore\` / \`ignore_only\` / \`except\` entries under \`breaking:\`." >&2
    exit 1
    ;;
  100)
    # buf's exit code for "breaking changes found": the exceptions are live.
    echo "$OUTPUT"
    echo "The exceptions are in use: the changes above are what they excuse."
    exit 0
    ;;
  *)
    # Anything else is buf failing to run (bad baseline, bad config), which
    # says nothing about staleness and must not read as a pass.
    echo "$OUTPUT" >&2
    echo "buf breaking did not run cleanly (exit $STATUS); cannot judge the exceptions." >&2
    exit 1
    ;;
esac
