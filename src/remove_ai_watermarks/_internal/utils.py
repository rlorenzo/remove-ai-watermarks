"""Small path helpers shared by optional image pipelines."""

from __future__ import annotations

import os
import secrets
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING

from remove_ai_watermarks._internal.constants import JPEG_SUFFIXES, SUPPORTED_FORMATS

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def is_supported_format(file_path: Path) -> bool:
    """Return whether ``file_path`` has a supported raster suffix."""
    return file_path.suffix.casefold() in SUPPORTED_FORMATS


def get_image_format(file_path: Path) -> str:
    """Return the Pillow save format used by the legacy metadata API.

    The metadata writer only has specialized PNG and JPEG paths. Other accepted
    inputs therefore use its PNG fallback, matching the established API contract.
    """
    return "JPEG" if file_path.suffix.casefold() in JPEG_SUFFIXES else "PNG"


@contextmanager
def atomic_output(output: Path) -> Generator[Path]:
    """Yield a sibling temporary path; publish it onto ``output`` only on success.

    The caller writes the temporary path by name. On a clean exit it is fsynced, given
    the mode of the file it replaces, and moved with ``os.replace``, so an interrupted
    write (Ctrl-C, full disk, a raising encoder) never truncates an existing ``output``
    -- which matters for in-place rewrites, where ``output`` is the only copy. The
    temporary stays owner-only (``0o600``) while it is written, so a restricted image's
    pixels never sit in a world-readable sibling; just before publishing it takes the
    replaced file's mode, or ``0o666 & ~umask`` for a new output (what a plain write
    would give it). A symlinked ``output`` is published onto its target, as a plain
    write would be. The temporary is removed on failure.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        output = output.resolve()
    temporary = output.with_name(f".{output.stem}-{secrets.token_hex(6)}{output.suffix}")
    # Owner-only from creation: a broader mode narrowed later would let another local
    # user open the inode in between and keep reading through that descriptor.
    os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    try:
        yield temporary
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        if output.exists():
            shutil.copymode(output, temporary)
        else:
            os.chmod(temporary, _new_file_mode(output.parent))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _new_file_mode(directory: Path) -> int:
    """Return ``0o666 & ~umask`` by creating an empty probe that never holds data.

    ``os.umask`` can only be read by setting it, which races other threads.
    """
    probe = directory / f".mode-probe-{secrets.token_hex(6)}"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        return os.fstat(fd).st_mode & 0o777
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
