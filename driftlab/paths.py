"""Path formatting for values persisted in DriftLab artifacts.

Run directories can be supplied to the runner and reporting tools either as
relative paths or as absolute paths.  JSON artifacts should not retain the
machine-specific form, so this module is the single place where filesystem
paths become portable strings.
"""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: str | os.PathLike) -> str:
    """Return *path* as a POSIX, repository-relative string.

    Relative inputs stay relative.  Absolute inputs inside the repository are
    made relative to the repository root; paths outside it are also rendered
    relative rather than leaking a host-specific absolute prefix.
    """
    value = Path(os.fspath(path))
    if not value.is_absolute():
        return value.as_posix()

    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(value, ROOT)).as_posix()
