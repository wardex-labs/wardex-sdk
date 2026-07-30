"""Reserved for a Python-side pipeline stage; empty on purpose.

The one pipeline stage that exists — PII masking — runs in the Rust core
(`crates/wardex-pipeline`), which is where byte-exact and hot-path work
belongs, so nothing imports this package today.
"""
