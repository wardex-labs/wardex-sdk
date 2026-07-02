import hashlib

from wardex_sdk import _hash


def test_object_keys_sorted_and_compact():
    out = _hash.canonicalize({"b": 2, "a": 1})
    assert out == b'{"a":1,"b":2}'


def test_key_order_independence_same_hash():
    h1 = _hash.hash_canonical({"x": 1, "y": [1, 2, 3]})
    h2 = _hash.hash_canonical({"y": [1, 2, 3], "x": 1})
    assert h1 == h2


def test_nested_and_array_order_preserved():
    out = _hash.canonicalize({"arr": [3, 1, 2], "obj": {"z": 1, "a": 2}})
    assert out == b'{"arr":[3,1,2],"obj":{"a":2,"z":1}}'


def test_string_unicode_not_escaped():
    out = _hash.canonicalize({"k": "café"})
    assert out == '{"k":"café"}'.encode()


def test_float_practical_range():
    assert _hash.canonicalize(0.7) == b"0.7"
    assert _hash.canonicalize(1.0) == b"1"
    assert _hash.canonicalize(100) == b"100"
    assert _hash.canonicalize(-0.0) == b"0"


def test_bool_and_null():
    assert _hash.canonicalize(True) == b"true"
    assert _hash.canonicalize(None) == b"null"


def test_sha256_hex_matches_hashlib():
    data = b'{"a":1}'
    assert _hash.sha256_hex(data) == hashlib.sha256(data).hexdigest()


def test_hash_canonical_is_sha256_of_canonical():
    obj = {"a": 1, "b": "x"}
    expected = hashlib.sha256(_hash.canonicalize(obj)).hexdigest()
    assert _hash.hash_canonical(obj) == expected
