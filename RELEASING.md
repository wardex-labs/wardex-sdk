# Releasing

Releases are per-package, triggered by a package-scoped git tag.

## Python (`wardex-sdk`)

```bash
python scripts/release.py python 0.1.0b2   # bumps version, rolls CHANGELOG, commits, tags
git push origin main --tags                # triggers release-python.yml
```

The `release-python.yml` workflow (on `python-v*` tags) builds abi3 wheels
(manylinux x86_64/aarch64, macOS x86_64/arm64, Windows) plus an sdist, and
publishes to PyPI via Trusted Publishing (OIDC) — no API token.

Version scheme (PEP 440): `0.1.0b1` → `0.1.0b2` → `0.1.0rc1` → `0.1.0` → `0.1.1` / `0.2.0`.

## Future SDKs

Node and Java will use their own scoped tags (`js-v*`, `java-v*`) and their own
release workflows, independent of Python.
