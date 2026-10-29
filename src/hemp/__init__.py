"""HEMP: Help Evaluate My Prompt."""

from importlib.metadata import PackageNotFoundError, version

from hemp.dataset import DatasetError, load_dataset
from hemp.profile import DatasetProfile, profile_dataset, profile_rows

try:
    __version__ = version("hemp-eval")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = [
    "DatasetError",
    "DatasetProfile",
    "__version__",
    "load_dataset",
    "profile_dataset",
    "profile_rows",
]
