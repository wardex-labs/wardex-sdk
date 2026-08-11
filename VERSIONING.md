# Versioning

How wardex SDKs and the wire schema are versioned, and what each version
number promises. This is policy, recorded once; the CHANGELOG records what
actually changed.

## Version semantics

- Versions follow **PEP 440**, with **semver semantics from 1.0.0**: additive
  changes bump the minor version, breaking changes to any public name or
  documented behavior bump the major version, fixes bump the patch version.
- **0.x betas may break public API between minor versions.** Every break is
  announced in the CHANGELOG entry of the release that ships it — never
  silently.

## Deprecation (from 1.0.0)

- A **renamed or removed Python name** keeps working for **at least one minor
  version** behind a module-level `__getattr__` that emits a
  `DeprecationWarning` naming the replacement.
- A **moved or reshaped config spelling** is different: it is refused with a
  `TypeError` naming its new home, immediately and in every version. A config
  setting is never silently ignored — silence there is indistinguishable from
  the setting being honoured.

## One contract, several SDKs

- The Python, Node, and Java SDKs **version independently** — a release of one
  does not require a release of the others.
- What they share is the **cross-language contract**: the config group names
  (`backend`, `pii`, `batching`, `limits`, `propagation`, `adapters`), the
  `WARDEX_*` environment variable names, and the wire vocabulary. Type names
  are byte-identical across languages; field names map mechanically
  (snake_case to camelCase in Node); duration fields are float **seconds**
  everywhere.

## The wire schema

- `proto/wardex/v1` evolves **append-only**: fields and enum values are
  added, never renumbered and never reused. Anything the schema ever shipped
  stays decodable.
- The schema versions with its **package name** (`wardex.v1`), independently
  of every SDK version. A breaking wire change means a new package
  (`wardex.v2`), not a mutation of v1.

## Probing the version

- `__version__` is the canonical runtime version in every SDK
  (`wardex_sdk.__version__` in Python).
