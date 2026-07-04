"""Smoke: verifies wardex_sdk package import + native module loading."""

from importlib.metadata import version


def test_import_wardex_sdk() -> None:
    import wardex_sdk

    # __version__ is derived from installed package metadata, so it must match the
    # distribution version and must not fall back to the source-tree default. This
    # assertion needs no per-release update.
    assert wardex_sdk.__version__ == version("wardex-sdk")
    assert wardex_sdk.__version__ != "0.0.0.dev0"


def test_native_module_loads() -> None:
    import wardex_sdk

    assert hasattr(wardex_sdk, "_wardex_native")
    assert wardex_sdk._wardex_native.__version__
