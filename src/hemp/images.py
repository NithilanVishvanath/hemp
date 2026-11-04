"""Image columns: cells holding image file paths, URLs, or data URIs.

A column counts as an image column when every non-empty value looks like an
image reference. Relative paths are resolved against the dataset file's folder.
The model receives the image itself, not the path.
"""

import base64
import hashlib
import mimetypes
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hemp.dataset import Row

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_DATA_URI_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)


class ImageError(ValueError):
    """An image can't be found or read."""


def looks_like_image(value: object) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s or "\n" in s:
        return False
    if s.startswith("data:image/"):
        return True
    path = urlparse(s).path if s.startswith(("http://", "https://")) else s
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def image_columns(rows: Sequence[Row], columns: Iterable[str] | None = None) -> set[str]:
    """Columns whose every non-empty value is an image reference."""
    names = (
        list(columns) if columns is not None else list(dict.fromkeys(k for r in rows for k in r))
    )
    found = set()
    for name in names:
        values = [r.get(name) for r in rows if r.get(name) not in (None, "")]
        if values and all(looks_like_image(v) for v in values):
            found.add(name)
    return found


@dataclass(frozen=True)
class ImageRef:
    """An image as written in the dataset, and where it actually is."""

    source: str  # the cell value
    location: str  # absolute file path, URL, or data URI

    @property
    def is_file(self) -> bool:
        return not self.location.startswith(("http://", "https://", "data:"))

    @property
    def media_type(self) -> str:
        match = _DATA_URI_RE.match(self.location)
        if match:
            return match.group(1)
        path = urlparse(self.location).path if not self.is_file else self.location
        guessed, _ = mimetypes.guess_type(path)
        return guessed or "image/png"

    def base64_data(self) -> str:
        """The image bytes, base64-encoded. Only for files and data URIs."""
        match = _DATA_URI_RE.match(self.location)
        if match:
            return match.group(2)
        try:
            return base64.b64encode(Path(self.location).read_bytes()).decode("ascii")
        except OSError as e:
            raise ImageError(f"Can't read image {self.source}: {e.strerror or e}") from None


def resolve(value: str, base_dir: Path | None) -> ImageRef:
    s = value.strip()
    if s.startswith(("http://", "https://", "data:")):
        return ImageRef(source=value, location=s)
    path = Path(s)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return ImageRef(source=value, location=str(path.resolve()))


def missing_images(rows: Sequence[Row], columns: Iterable[str], base_dir: Path | None) -> list[str]:
    """Image files referenced by the rows that don't exist (URLs aren't checked)."""
    missing = []
    for row in rows:
        for name in columns:
            value = row.get(name)
            if isinstance(value, str) and value.strip():
                ref = resolve(value, base_dir)
                if ref.is_file and not Path(ref.location).is_file():
                    missing.append(value)
    return list(dict.fromkeys(missing))


def images_digest(rows: Sequence[Row], columns: Iterable[str], base_dir: Path | None) -> str:
    """Hash of the image contents, so replacing an image file changes the dataset hash."""
    digest = hashlib.sha256()
    for row in rows:
        for name in sorted(columns):
            value = row.get(name)
            if not isinstance(value, str) or not value.strip():
                continue
            ref = resolve(value, base_dir)
            digest.update(ref.source.encode())
            if ref.is_file:
                try:
                    digest.update(hashlib.sha256(Path(ref.location).read_bytes()).digest())
                except OSError:
                    digest.update(b"missing")
    return digest.hexdigest()
