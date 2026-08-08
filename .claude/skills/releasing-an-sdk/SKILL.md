---
name: releasing-an-sdk
description: >-
  Use this to publish a new version of the wardex Python SDK to PyPI. This is the
  release skill: reach for it whenever the user wants to ship, cut, publish, roll
  out, or release a version of the SDK or Python package; bump the version and
  publish; get a build onto PyPI; put out the next beta, rc, or final; or retry a
  release that never landed. It covers direct asks ("publish the sdk to pypi",
  "ship 0.1.0b3", "cut the next beta") and situational ones ("we merged the fix,
  time to push a new beta to users"). It runs preflight checks, selects the
  version, drafts the changelog, and gates the single irreversible tag-push behind
  your confirmation — so always release through it rather than hand-running git
  tag/push. Don't use it for editing __version__ on its own, tagging unrelated
  commits, writing blog or launch notes, or one-time PyPI/OIDC account setup.
---

# Releasing a wardex SDK

## What this is

A thin orchestration layer over the release assets that already exist in this
repo — `scripts/release.py`, `.github/workflows/release-python.yml`,
`RELEASING.md`. It does not invent a new release mechanism; it drives the
existing one safely.

**The core principle that shapes everything here:** reversible work is
automated, one irreversible action is gated. Preflight, version bump, changelog,
the release commit, and the tag *creation* are all local and undoable — do them
freely. Pushing the tag is the single irreversible act: it triggers a PyPI
publish, and a published version can never be changed or reused. So the whole
skill funnels toward one human confirmation before that push, and nothing else
needs to interrupt the user.

## Scope (be honest about what works today)

- **Python only.** `scripts/release.py` currently accepts only `python` as its
  first argument. The tag scheme (`python-v*`) and workflow split are designed so
  `js-v*` / `java-v*` can slot in later, but those SDKs don't exist yet. If asked
  to release a non-Python SDK, say it isn't wired up yet rather than improvising.
- **No standalone Rust-core release.** The Rust core is statically compiled into
  each SDK's wheel (path dependency + maturin), not published as its own
  artifact. A core fix reaches users only by cutting a new SDK version that
  bundles it. There is no "release the core" path to look for.

## The gate rule (read this before doing anything)

The tag push publishes to PyPI **permanently**. Never push a release tag without
an explicit, unambiguous go-ahead from the human in this conversation — showing
them the plan and getting a "yes" is the point of the whole skill. If you are
running without a human who can confirm (an automated or evaluation context),
**stop at the gate and print the plan; do not push.** This is what makes the
skill safe to rehearse.

## The flow

Work through these in order. Stop and report if any preflight check fails —
don't paper over a red check.

### 1. Preflight (hard guards)

These exist because a release inherits whatever state `main` is in. If `main` is
broken or unpushed, the release ships broken or unreproducible bits. Verify all
of:

```bash
git rev-parse --abbrev-ref HEAD          # must be "main"
git status --porcelain                    # must be empty (clean tree)
git fetch origin
git rev-parse HEAD; git rev-parse origin/main   # must be equal (in sync)
```

CI green on the exact commit you're releasing (this is what lets us skip
re-running tests locally — the passing CI run already tested this code):

```bash
gh run list --workflow=ci.yml --branch=main --limit=20 \
  --json headSha,status,conclusion
# find the row whose headSha == current HEAD; require conclusion == "success"
```

Target version not already on PyPI (a taken version is burned and cannot be
reused — see Failure recovery):

```bash
curl -s -o /dev/null -w "%{http_code}" \
  https://pypi.org/pypi/wardex-sdk/<version>/json    # 404 = free, 200 = taken
```

Changelog is ready to roll:

```bash
grep -q '## \[Unreleased\]' CHANGELOG.md   # release.py aborts without this header
```

Rehearse the wheel gate if anything under it changed since the last release —
the release workflow, the build or smoke matrix, the Rust core, or a dependency
the wheel links against. `release-python.yml` also takes a `workflow_dispatch`,
and `publish` requires a pushed tag, so a dispatched run builds and smokes all
five wheels and stops short of PyPI whatever ref you aim it at:

```bash
gh workflow run release-python.yml --ref main
gh run list --workflow=release-python.yml --limit=1 --json databaseId,status
gh run watch <databaseId>   # all six smoke entries green
```

Skip it for a release that only touches Python or Rust source CI already
covered. Run it when in doubt: the alternative place to discover that the
Rosetta probe or the QEMU container broke is under a tag you cannot take back.

### 2. Version selection

Read the current version:

```bash
grep -m1 '^version' sdks/python/pyproject.toml
```

**Before proposing a bump, check whether the current version is already
published** — this is the step naive releases get wrong:

```bash
curl -s -o /dev/null -w "%{http_code}" \
  https://pypi.org/pypi/wardex-sdk/<current>/json    # 404 = never shipped
```

- **Current version is NOT on PyPI (404):** a prior release never completed. The
  target is the **current version itself** — you're finishing that release, not
  bumping. If a `python-v<current>` tag already exists (a build that failed after
  tagging), this is the re-tag case in Failure recovery, not a new version. Do
  not propose the next number just because a version string sits in pyproject.
  **In this case skip steps 3 and 4 (changelog draft and release.py) entirely** —
  the release commit, the tag, and this version's changelog section already
  exist, so re-running release.py would double-roll the changelog and try to
  re-bump an already-current version. Go straight to the re-tag sequence in
  Failure recovery: fix the cause, delete and recreate the tag at the fixed
  commit, then the gated push.
- **Current version IS on PyPI (200):** it shipped. Now propose the next version
  per the progression below and let the user pick.

PEP 440 pre-release progression, for proposing the next step:

```
0.1.0b1 → 0.1.0b2 → 0.1.0rc1 → 0.1.0 → 0.1.1 / 0.2.0
         (bN beta)  (rcN rc)   (final) (patch / minor)
```

Suffixes: `aN` alpha, `bN` beta, `rcN` release candidate — pre-releases sort
*before* the final (`0.1.0b1` < `0.1.0`). Propose the natural next version, but
let the user override (e.g., jumping beta → rc, or beta → final). Validate the
string is PEP 440-shaped before continuing.

### 3. Draft the CHANGELOG from the diff

The point is to save the user from writing release notes by hand, using the fact
that this repo mandates Conventional Commits — so each commit subject already
carries a human-summarized intent and a type.

Collect commits since the last release and group them:

```bash
git log python-v<prev>..HEAD --pretty=format:'%s'
```

Map Conventional types to Keep a Changelog sections:

| Commit prefix | CHANGELOG section |
|---|---|
| `feat:` | Added |
| `fix:` | Fixed |
| `refactor:` / `perf:` | Changed |
| `chore` / `ci` / `docs` / `test` / `style` | omit by default (internal, not user-facing) |

Write the grouped draft **under the existing `## [Unreleased]` header** in
`CHANGELOG.md`, then let the user curate it — drop internal noise, reword for
users, reorder. Two things to respect:

- The source is the commit subjects, not the raw code diff. Read the diff only
  if a subject is too terse to classify — summarizing from the diff directly just
  adds noise.
- If `## [Unreleased]` already has hand-written entries, **preserve them and
  merge** — propose additions, never overwrite the user's words.

release.py (next step) turns this `## [Unreleased]` content into the versioned
section, so you don't write the version header yourself.

### 4. Prep (release.py — still fully reversible)

Skip this step if step 2 found you're **finishing an already-prepared version**
(current version unpublished, its tag and changelog section already in place) —
go straight to the re-tag path in Failure recovery. Otherwise, for a genuinely
new version:

```bash
python scripts/release.py python <version>
```

This bumps `pyproject.toml`, rolls the `## [Unreleased]` content under a new
`## [<version>] - <date>` header, commits as `chore(release): python-v<version>`,
and creates the tag `python-v<version>`. It does **not** push. Then show the user
exactly what will ship:

```bash
git show --stat HEAD          # the release commit
git tag --points-at HEAD      # the tag that will trigger publish
```

### 5. ✋ Gate — the one confirmation

Tell the user plainly: "Pushing this tag publishes wardex-sdk `<version>` to PyPI
permanently. Push now?" Wait for an explicit yes. This is the only place the
skill blocks the user, and it's deliberate — everything before it can be undone,
nothing after it can.

### 6. Publish (only after the yes)

```bash
git push origin main
git push origin python-v<version>   # push the specific tag — NOT --tags
```

### 7. Monitor

The tag push started `release-python.yml`. Watch it and confirm the outcome
rather than assuming success:

```bash
gh run list --workflow=release-python.yml --limit=1 --json databaseId,status
gh run watch <databaseId>
# then confirm the version actually went live:
curl -s -o /dev/null -w "%{http_code}" \
  https://pypi.org/pypi/wardex-sdk/<version>/json    # expect 200
```

The run is three stages and the publish is last: `build wheels (...)` produces
the five abi3 wheels, `smoke-test wheel (...)` installs and imports every one of
them (six entries — each wheel on 3.12, plus linux x86_64 again on the 3.10
floor, which is the only entry that can see an abi3 tag that stopped covering
the floor), and `publish` uploads only if both stages are fully green. So a red
smoke entry is a real platform result, not a known blind spot.

If the run fails, go to Failure recovery.

## Failure recovery

Everything turns on one question: **did PyPI actually receive the version?** PyPI
burns a version name only when files are uploaded. So there are two worlds with
opposite fixes.

### The build or the smoke test failed → version is still free → re-tag the SAME version

If a `build (...)` **or** a `smoke-test wheel (...)` job failed, the `publish`
job was skipped (`needs: [build, smoke]` unmet), so nothing uploaded and
`<version>` is still available. This is the common case: two gates guard the
publish, and either one red leaves the version free. Check both before
concluding anything — every `build (...)` green with one smoke entry red is
this world, not the burned one below, and the smoke matrix is `fail-fast: false`
precisely so the red entry names the platform.

Either failure is almost always a *build/config* bug, not a test bug (tests
already passed in CI): a build job says the wheel could not be produced, a smoke
entry says it was produced but does not install or import on that platform. Fix
it in the repo files (workflow YAML, Cargo/pyproject, source), commit, rehearse
the fix with a dispatched run (see Preflight) since a re-tag is another
irreversible push, then re-tag:

```bash
git push origin :python-v<version>   # delete the remote tag
git tag -d python-v<version>          # delete the local tag
# ... commit the fix ...
git tag python-v<version>            # re-create at the FIXED commit
git push origin python-v<version>    # re-trigger
```

The re-tag **must** point at the fixed commit: a tag-triggered workflow runs the
workflow file *as it exists at the tagged commit*. If you fixed the workflow but
re-tagged the old commit, it would run the broken workflow again. (This is
exactly what happened re-releasing `0.1.0b1`.)

### The publish succeeded (even partially) → version is burned → bump

If `publish` ran and uploaded anything, `<version>` exists on PyPI forever and
cannot be reused. Don't fight it — start the flow again for the next version
(e.g., `0.1.0b2 → 0.1.0b3`). Preflight's PyPI check will confirm the old version
is taken and the new one is free.

### Failure before the push (preflight or prep)

Nothing was pushed, so PyPI knows nothing — it's fully local. Fix the cause
(red CI, dirty tree, missing `## [Unreleased]`) and restart the flow. If
release.py already created a local commit and tag but you caught it before
pushing, just delete them locally (`git tag -d python-v<version>` and reset the
commit) — there's nothing to undo remotely.

## Out of scope (don't do these here)

- **First-time PyPI/OIDC setup** (Trusted Publisher, org creation, project
  transfer) — one-time admin, already done. Assume it exists.
- **js/java publish** — not wired up; say so rather than improvising.
- **Publishing an sdist** — the workflow ships wheels only (abi3, five
  platforms). `pip install` on anything outside that set has no source fallback
  and fails. Adding one is a workflow change, not this skill's job.
