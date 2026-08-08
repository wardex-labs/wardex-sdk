# Contributing

Thanks for your interest in wardex-sdk.

## Development setup

```bash
uv sync                 # builds the native module
cargo test --workspace
uv run pytest sdks/python/tests -v
```

## Before opening a PR

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check sdks/python
uv run ruff format --check sdks/python
buf lint
scripts/check-py310.sh   # if you touched anything under sdks/python
```

`scripts/check-py310.sh` runs the suite on CPython 3.10, the floor
`sdks/python/pyproject.toml` declares. The development venv is a much newer
interpreter, so a 3.11+-only API passes locally and turns red only in CI's 3.10
job — which imports the tests as well as the source, so run the check for a
change to either. The script builds its own `.venv-py310` and leaves your
`.venv`'s interpreter and dependency set alone; the one thing the two share is
the editable `_wardex_native.abi3.so` under `sdks/python/src`, which the check
rebuilds (harmless — the extension is abi3 — but worth knowing if you are
debugging the native module).

## Conventions

- Comments, docstrings, and prose are **English**.
- Commit messages follow Conventional Commits.
- Keep the Rust core domain-agnostic; put byte-exact/hot-path work in Rust.

No CLA is required for the beta.
