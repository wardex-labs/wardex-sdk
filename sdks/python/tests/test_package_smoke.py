"""Smoke: verifies wardex_sdk package import + native module loading."""


def test_import_wardex_sdk() -> None:
    import wardex_sdk

    assert wardex_sdk.__version__ == "0.1.0"


def test_native_module_loads() -> None:
    import wardex_sdk

    assert hasattr(wardex_sdk, "_wardex_native")
    assert wardex_sdk._wardex_native.__version__
