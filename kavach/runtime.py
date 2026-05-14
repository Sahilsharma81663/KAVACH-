from __future__ import annotations

import importlib
import re
import sys
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / ".vendor"


def _current_extension_tags() -> set[str]:
    major, minor = sys.version_info[:2]
    tags = {
        f"cp{major}{minor}",
        f"cpython-{major}{minor}",
    }
    for suffix in EXTENSION_SUFFIXES:
        tags.update(re.findall(r"(cp\d{2,3}|cpython-\d{2,3})", suffix.lower()))
    return tags


@lru_cache(maxsize=1)
def vendor_status() -> tuple[bool, str]:
    if not VENDOR_DIR.exists():
        return False, "Vendor directory not found."

    compiled_files = list(VENDOR_DIR.rglob("*.pyd")) + list(VENDOR_DIR.rglob("*.so"))
    if not compiled_files:
        return True, "Vendor directory contains pure Python packages only."

    vendor_tags: set[str] = set()
    for file_path in compiled_files[:200]:
        vendor_tags.update(re.findall(r"(cp\d{2,3}|cpython-\d{2,3})", file_path.name.lower()))

    if not vendor_tags:
        return True, "Vendor directory does not expose tagged native extensions."

    current_tags = _current_extension_tags()
    if current_tags.intersection(vendor_tags):
        return True, "Vendor directory matches the running interpreter."

    return (
        False,
        f"Vendor extensions target {sorted(vendor_tags)}, but this interpreter exposes {sorted(current_tags)}.",
    )


def vendor_is_compatible() -> bool:
    compatible, _ = vendor_status()
    return compatible


def vendor_message() -> str:
    return vendor_status()[1]


def _clear_partial_import(module_name: str) -> None:
    prefix = f"{module_name}."
    stale_modules = [
        loaded_name
        for loaded_name in list(sys.modules)
        if loaded_name == module_name or loaded_name.startswith(prefix)
    ]
    for loaded_name in stale_modules:
        sys.modules.pop(loaded_name, None)


def import_optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as primary_error:
        compatible, _ = vendor_status()
        if not compatible or not VENDOR_DIR.exists():
            raise primary_error

        inserted = False
        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
            inserted = True

        _clear_partial_import(module_name)
        try:
            return importlib.import_module(module_name)
        except Exception:
            if inserted and vendor_path in sys.path:
                sys.path.remove(vendor_path)
            raise

