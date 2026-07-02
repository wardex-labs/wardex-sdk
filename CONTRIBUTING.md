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
```

## Conventions

- Comments, docstrings, and prose are **English**.
- Commit messages follow Conventional Commits.
- Keep the Rust core domain-agnostic; put byte-exact/hot-path work in Rust.

No CLA is required for the beta.
