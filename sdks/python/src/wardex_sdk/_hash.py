"""RFC 8785 JSON Canonicalization Scheme (JCS) + SHA-256.

This normalization rule is the contract any reimplementation in the Rust core must
reproduce byte-for-byte. Limitation: ECMAScript Number→String exponential notation
(very large/small values) is not implemented — only accurate for the practical
decimal range (LLM parameters). A ryu-based approach would close that gap if the
extreme range ever reaches this function.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _number(value: float | int) -> str:
    # bool is branched off before this is called (isinstance(bool) checked first)
    if isinstance(value, int):
        return str(value)
    f = float(value)
    if f == 0.0:  # includes -0.0 → "0"
        return "0"
    if f == int(f) and abs(f) < 1e16:
        return str(int(f))  # 1.0 -> "1"
    # shortest round-trip (Python repr) — matches ES6 in the practical decimal range
    return repr(f)


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        # json.dumps escaping is compatible with RFC 8785 §3.2.2.2 (lowercase \u, minimal escaping)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    if isinstance(value, dict):
        # sort keys by UTF-16 code unit order (RFC 8785 §3.2.3)
        items = sorted(value.items(), key=lambda kv: str(kv[0]).encode("utf-16-be"))
        return (
            "{"
            + ",".join(
                json.dumps(str(k), ensure_ascii=False) + ":" + _serialize(v) for k, v in items
            )
            + "}"
        )
    raise TypeError(f"Type not serializable by JCS: {type(value)!r}")


def canonicalize(obj: Any) -> bytes:
    return _serialize(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_canonical(obj: Any) -> str:
    return sha256_hex(canonicalize(obj))
