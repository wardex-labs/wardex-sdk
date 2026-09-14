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

The `release-python.yml` workflow (on `python-v*` tags) runs three gates in
parallel and publishes only when all three are green: `build` makes the abi3
wheels (manylinux x86_64/aarch64, macOS x86_64/arm64, Windows); `smoke`
installs and imports each one; `gate` re-runs every check CI runs — `cargo fmt
--check`, `clippy -D warnings`, `cargo test --workspace`, `ruff check`, `ruff
format --check`, and the full Python suite — on the exact commit the tag points
at, on the floor interpreter and the newest. `publish` then uploads to PyPI via
Trusted Publishing (OIDC) — no API token. (A source distribution is a
follow-up; the beta ships wheels only.)

The same workflow also runs on demand, which is how you prove the build and
wheel-install steps *before* the tag exists:

```bash
gh workflow run release-python.yml --ref <branch>   # builds + installs every wheel
```

Publishing requires a pushed `python-v*` tag, so a manual run stops after the
install step whatever ref you point it at. Use it whenever the workflow, the
build matrix, or anything the wheel links against changed — those paths are
otherwise first exercised by a tag push you cannot take back.

Version scheme (PEP 440): `0.1.0b1` → `0.1.0b2` → `0.1.0rc1` → `0.1.0` → `0.1.1` / `0.2.0`.

Suffixes: `aN` alpha, `bN` beta, `rcN` release candidate — pre-releases sort
before the final version (`0.1.0b1` < `0.1.0`). Note: `pip install wardex-sdk`
skips pre-releases by default; testers need `pip install wardex-sdk --pre`
(or an exact pin like `wardex-sdk==0.1.0b1`).

## Notes

- **A tag triggers the release, not a branch push.** Pushing `main` alone runs CI
  but does not publish; the `python-v*` tag is what starts `release-python.yml`.
  Push the specific tag (`git push origin python-vX.Y.Z`), not `--tags`. The
  workflow can also be started by hand from any branch (see above) — that path
  builds and installs the wheels but never publishes.
- **A published version is immutable.** PyPI will not let you reuse or overwrite a
  version. If a release is broken, publish the next one (e.g. `0.1.0b3`).
- **Publishing is all-or-nothing.** The publish job needs every platform wheel to
  build, every smoke entry to install and import it, *and* both `gate` entries
  to pass; if any stage has one red job, nothing is published (no partial
  release) and the version is still free to re-tag once the cause is fixed.
- **The tag commit is tested at publish time, not trusted.** `ci.yml` runs on
  branch pushes and pull requests; `release-python.yml` runs on `python-v*`
  tags; the two do not know about each other. So the release workflow carries
  its own `gate` job with the same checks, run on the tagged commit itself. A
  red `gate` on a commit whose CI was green means the tag does not point where
  you think it does — check `git rev-parse python-vX.Y.Z` against `origin/main`
  before anything else. What no gate covers: a tag pushed by hand around
  `scripts/release.py` still publishes if the checks pass; the script is the
  procedure, the workflow is the guard.

## Future SDKs

Node and Java will use their own scoped tags (`js-v*`, `java-v*`) and their own
release workflows, independent of Python.
