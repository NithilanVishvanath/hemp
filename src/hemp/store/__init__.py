"""SQLite result store (run diffing comes in milestone 7)."""

from hemp.store.db import Store, StoreError, default_db_path

__all__ = ["Store", "StoreError", "default_db_path"]
