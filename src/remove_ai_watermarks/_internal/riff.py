"""Bounded AVI metadata reader for native TC260 AIGC labels.

TC260-PG-20257A stores the label in an AVI ``LIST/INFO`` chunk whose child
chunk ID is ``AIGC`` and whose value is the normative JSON object. The walker
seeks over media chunks and reads only bounded metadata values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from remove_ai_watermarks.metadata import MAX_TC260_VALUE_BYTES, parse_tc260_aigc_json


def _iter_chunks(stream: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(chunk_id, size, payload_start)`` for each RIFF chunk in ``[start, end)``.

    Stops at a truncated header or a chunk that overruns ``end``. The stream is left
    positioned at the payload start, so the caller can read the value directly.
    """
    position = start
    while position + 8 <= end:
        stream.seek(position)
        chunk_id = stream.read(4)
        size_raw = stream.read(4)
        if len(chunk_id) != 4 or len(size_raw) != 4:
            return
        size = int.from_bytes(size_raw, "little")
        payload_start = position + 8
        payload_end = payload_start + size
        if payload_end > end:
            return
        yield chunk_id, size, payload_start
        position = payload_end + (size & 1)


def _info_payloads(
    stream: BinaryIO,
    start: int,
    end: int,
) -> tuple[bytes, ...]:
    found: list[bytes] = []
    for chunk_id, size, _ in _iter_chunks(stream, start, end):
        if chunk_id == b"AIGC" and size <= MAX_TC260_VALUE_BYTES:
            value = stream.read(size)
            if len(value) == size and parse_tc260_aigc_json(value) is not None:
                found.append(value.rstrip(b"\x00 "))
    return tuple(found)


def tc260_aigc_payloads(path: str | Path) -> tuple[bytes, ...]:
    """Read validated TC260 values from an AVI ``LIST/INFO/AIGC`` chunk."""
    found: list[bytes] = []
    try:
        with open(path, "rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"AVI ":
                return ()
            stream.seek(0, 2)
            file_size = stream.tell()
            declared_end = min(8 + int.from_bytes(header[4:8], "little"), file_size)
            for chunk_id, size, payload_start in _iter_chunks(stream, 12, declared_end):
                if chunk_id == b"LIST" and size >= 4 and stream.read(4) == b"INFO":
                    found.extend(_info_payloads(stream, payload_start + 4, payload_start + size))
    except OSError:
        return ()
    return tuple(found)
