"""Small path helpers shared by optional image pipelines."""

from __future__ import annotations

import os
import secrets
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING

from remove_ai_watermarks._internal.constants import SUPPORTED_FORMATS

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

_PIL_FORMAT_BY_SUFFIX = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}


def is_supported_format(file_path: Path) -> bool:
    """Return whether ``file_path`` has a supported raster suffix."""
    return file_path.suffix.casefold() in SUPPORTED_FORMATS


def get_image_format(file_path: Path) -> str:
    """Return the Pillow save format used by the legacy metadata API.

    The metadata writer only has specialized PNG and JPEG paths. Other accepted
    inputs therefore use its PNG fallback, matching the established API contract.
    """
    return _PIL_FORMAT_BY_SUFFIX.get(file_path.suffix.casefold(), "PNG")


@contextmanager
def atomic_output(output: Path) -> Generator[Path]:
    """Yield a sibling temporary path; publish it onto ``output`` only on success.

    The caller writes the temporary path by name. On a clean exit it is fsynced, given
    the mode of the file it replaces, and moved with ``os.replace``, so an interrupted
    write (Ctrl-C, full disk, a raising encoder) never truncates an existing ``output``
    -- which matters for in-place rewrites, where ``output`` is the only copy. The
    temporary is created ``0o666 & ~umask`` (not tempfile's ``0o600``), so a new
    output gets the same mode a plain write would give it. It is removed on failure.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}-{secrets.token_hex(6)}{output.suffix}")
    os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    try:
        yield temporary
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        if output.exists():
            shutil.copymode(output, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
