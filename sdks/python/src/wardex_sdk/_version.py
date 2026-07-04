"""Single source of truth for the SDK version.

The version is read from the installed package metadata, which maturin populates
from ``pyproject.toml`` at build time. Deriving it here means the runtime version
(``wardex_sdk.__version__`` and the ``SdkInfo`` reported in telemetry) can never
drift from the distribution version — there is no hardcoded string to update at
release time.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wardex-sdk")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0.dev0"
