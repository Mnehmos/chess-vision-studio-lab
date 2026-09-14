import json

import pytest

from cvslab.hashing import canonical_json, hash_obj, sha256_bytes


def test_canonical_json_is_key_sorted_and_compact():
    assert canonical_json({"b": 1, "a": [2, 3]}) == b'{"a":[2,3],"b":1}'


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_hash_is_stable_across_key_order():
    assert hash_obj({"a": 1, "b": 2}) == hash_obj({"b": 2, "a": 1})
    assert hash_obj({"a": 1}).startswith("sha256:")


def test_hash_distinguishes_int_and_float():
    assert hash_obj(1) != hash_obj(1.0)


def test_sha256_bytes_hex():
    digest = sha256_bytes(b"")
    assert digest == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert json.dumps({"h": digest})  # digest is plain JSON-safe text
