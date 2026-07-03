# Releasing

Releases are per-package, triggered by a package-scoped git tag.

## Python (`wardex-sdk`)

```bash
python scripts/release.py python 0.1.0b2   # bumps version, rolls CHANGELOG, commits, tags — does NOT push
git push origin main                       # push the release commit
git push origin python-v0.1.0b2            # push the TAG — this triggers the release
```

`scripts/release.py` only creates the commit and tag; pushing is left to you. It
also requires a `## [Unreleased]` header in `CHANGELOG.md` (it moves that section
under the new version) and stops if the header is missing.

The `release-python.yml` workflow (on `python-v*` tags) builds abi3 wheels
(manylinux x86_64/aarch64, macOS x86_64/arm64, Windows) plus an sdist, and
publishes to PyPI via Trusted Publishing (OIDC) — no API token.

Version scheme (PEP 440): `0.1.0b1` → `0.1.0b2` → `0.1.0rc1` → `0.1.0` → `0.1.1` / `0.2.0`.

Suffixes: `aN` alpha, `bN` beta, `rcN` release candidate — pre-releases sort
before the final version (`0.1.0b1` < `0.1.0`). Note: `pip install wardex-sdk`
skips pre-releases by default; testers need `pip install wardex-sdk --pre`
(or an exact pin like `wardex-sdk==0.1.0b1`).

## Notes

- **A tag triggers the release, not a branch push.** Pushing `main` alone runs CI
  but does not publish; the `python-v*` tag is what starts `release-python.yml`.
  Push the specific tag (`git push origin python-vX.Y.Z`), not `--tags`.
- **A published version is immutable.** PyPI will not let you reuse or overwrite a
  version. If a release is broken, publish the next one (e.g. `0.1.0b3`).
- **Publishing is all-or-nothing.** The publish job needs every platform wheel and
  the sdist to build; if any fails, nothing is published (no partial release).
- **CI and release are independent.** `ci.yml` runs on branch pushes/PRs;
  `release-python.yml` runs on `python-v*` tags. A CI failure does not block a
  tagged release, and vice versa.

## Future SDKs

Node and Java will use their own scoped tags (`js-v*`, `java-v*`) and their own
release workflows, independent of Python.
