"""Tests for the GGUF metadata reader (parameter counting for the 500M cap)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tiny_tactician.models.gguf import parameter_count, read_gguf_metadata


def _write_minimal_gguf(path: Path) -> int:
    """Write a tiny valid GGUF with two tensors and return the expected count."""
    tensors = [(64, 32), (16, 8)]
    total = sum(a * b for a, b in tensors)

    def string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", len(tensors))  # tensor count
    body += struct.pack("<Q", 1)  # kv count
    body += string("general.architecture") + struct.pack("<I", 8) + string("llama")
    for name, (rows, cols) in zip(("tok.weight", "blk.0.weight"), tensors):
        body += string(name)
        body += struct.pack("<I", 2)  # n_dims
        body += struct.pack("<Q", rows) + struct.pack("<Q", cols)
        body += struct.pack("<I", 0)  # ggml type
        body += struct.pack("<Q", 0)  # data offset
    path.write_bytes(body)
    return total


def test_parameter_count_of_a_written_gguf(tmp_path):
    path = tmp_path / "tiny.gguf"
    expected = _write_minimal_gguf(path)
    assert parameter_count(path) == expected


def test_metadata_is_parsed(tmp_path):
    path = tmp_path / "tiny.gguf"
    _write_minimal_gguf(path)
    info = read_gguf_metadata(path)
    assert info["metadata"]["general.architecture"] == "llama"
    assert len(info["tensors"]) == 2
    assert info["tensors"][0]["name"] == "tok.weight"


def test_non_gguf_file_is_rejected(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOTGGUF" + b"\x00" * 32)
    with pytest.raises(ValueError):
        read_gguf_metadata(path)
