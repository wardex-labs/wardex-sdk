#!/usr/bin/env bash
# The single definition of the Rust gate. `.github/workflows/ci.yml`,
# `.github/workflows/release-python.yml` and AGENTS.md all run this script, so
# CI, the release gate and the documented local check cannot drift apart.
#
# Usage: scripts/check-rust.sh [lint] [test] [deps] [msrv]   (default: lint test)
#
#   lint  fmt, clippy with and without features, rustdoc with warnings denied
#   test  the workspace test suite and the release-mode PII perf guard
#   deps  cargo-deny (licenses, bans, sources; advisories are advisory on PRs)
#         and cargo-machete (unused dependencies); needs both tools installed
#   msrv  `cargo check` on the `rust-version` declared in Cargo.toml; installs
#         that toolchain with rustup if it is missing
#
# Every cargo command passes `--locked`: the committed Cargo.lock is the
# dependency graph the wheel ships, and it is the graph these checks judge.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then steps=(lint test); fi

run() { echo "+ $*" >&2; "$@"; }

for step in "${steps[@]}"; do
  case "$step" in
    lint)
      run cargo fmt --all --check
      # Twice: the only feature, `panic-injection`, gates test-only FFI code that
      # a default-features run never compiles, so it would never be linted.
      run cargo clippy --workspace --all-targets --locked -- -D warnings
      run cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
      # The crates are not published, so contributors are the only readers of
      # these docs and most items are crate-private; without the flag rustdoc
      # never checks them.
      RUSTDOCFLAGS="-D warnings" run cargo doc --workspace --no-deps --document-private-items --all-features --locked
      ;;
    test)
      run cargo test --workspace --locked
      run cargo test -p wardex-pipeline --release --locked --test pii_perf -- --ignored
      ;;
    deps)
      run cargo deny --locked --all-features check licenses bans sources
      # Advisory data changes independently of the code; a new entry should
      # not block an unrelated change. CI makes this step blocking only on the
      # scheduled run.
      if [ "${CHECK_RUST_ADVISORIES_BLOCKING:-0}" = "1" ]; then
        run cargo deny --locked --all-features check advisories
      else
        run cargo deny --locked --all-features check advisories || echo "advisories: non-blocking on this run" >&2
      fi
      run cargo machete
      ;;
    msrv)
      msrv="$(grep -E '^rust-version' Cargo.toml | head -1 | cut -d'"' -f2)"
      [ -n "$msrv" ] || { echo "rust-version not found in Cargo.toml" >&2; exit 2; }
      rustup toolchain list | grep -q "^$msrv" || run rustup toolchain install "$msrv" --profile minimal
      run cargo "+$msrv" check --workspace --all-targets --all-features --locked
      ;;
    *)
      echo "unknown step: $step (expected lint, test, deps or msrv)" >&2
      exit 2
      ;;
  esac
done
