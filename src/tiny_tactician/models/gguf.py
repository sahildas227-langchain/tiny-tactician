"""Minimal GGUF metadata reader.

Used to prove the on-device model is inside the 500M parameter budget: it sums
the element counts of every tensor in the GGUF container.  Pure stdlib, no
third-party dependency.
"""

from __future__ import annotations

import struct
from pathlib import Path

__all__ = ["read_gguf_metadata", "parameter_count"]

_GGUF_MAGIC = b"GGUF"

# GGUF metadata value type -> (struct format, is_array)
_SCALARS = {
    0: ("B", 1),   # uint8
    1: ("b", 1),   # int8
    2: ("H", 2),   # uint16
    3: ("h", 2),   # int16
    4: ("I", 4),   # uint32
    5: ("i", 4),   # int32
    6: ("f", 4),   # float32
    7: ("?", 1),   # bool
    10: ("q", 8),  # int64
    11: ("d", 8),  # float64
    12: ("Q", 8),  # uint64
}


def _read(fh, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, fh.read(size))[0]


def _read_string(fh) -> str:
    length = _read(fh, "Q")
    return fh.read(length).decode("utf-8", errors="replace")


def _read_value(fh, vtype: int):
    if vtype == 8:  # string
        return _read_string(fh)
    if vtype == 9:  # array
        elem_type = _read(fh, "I")
        count = _read(fh, "Q")
        if elem_type == 8:
            return [_read_string(fh) for _ in range(count)]
        fmt, size = _SCALARS[elem_type]
        data = fh.read(size * count)
        return list(struct.unpack("<" + fmt * count, data))
    fmt, _size = _SCALARS[vtype]
    return _read(fh, fmt)


def read_gguf_metadata(path: str | Path) -> dict:
    """Return ``{"metadata": {...}, "tensors": [...], "parameter_count": int}``."""
    with Path(path).open("rb") as fh:
        if fh.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(fh, "I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _read(fh, "Q")
        kv_count = _read(fh, "Q")
        metadata: dict = {}
        for _ in range(kv_count):
            key = _read_string(fh)
            vtype = _read(fh, "I")
            metadata[key] = _read_value(fh, vtype)
        tensors: list[dict] = []
        total_params = 0
        for _ in range(tensor_count):
            name = _read_string(fh)
            n_dims = _read(fh, "I")
            dims = [_read(fh, "Q") for _ in range(n_dims)]
            _read(fh, "I")  # ggml type
            _read(fh, "Q")  # data offset
            count = 1
            for d in dims:
                count *= d
            total_params += count
            tensors.append({"name": name, "dims": dims, "count": count})
    return {"metadata": metadata, "tensors": tensors, "parameter_count": total_params}


def parameter_count(path: str | Path) -> int:
    """Total number of scalar parameters stored in a GGUF file."""
    return read_gguf_metadata(path)["parameter_count"]
