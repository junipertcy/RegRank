#!/usr/bin/env python3
#
# regrank -- Regularized methods for efficient ranking in networks
#
# Copyright (C) 2023-2025 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from __future__ import annotations

import importlib
import pathlib
import platform
from typing import Any

# Version resolution: prefer installed package metadata; during local dev, optionally
# read pyproject.toml if available.
try:
    from importlib import metadata as importlib_metadata  # py3.8+
except ImportError:  # pragma: no cover
    importlib_metadata = None  # type: ignore[assignment]

import tomllib

__title__ = "regrank"
__description__ = (
    "Regularized methods for efficient ranking in networks (SpringRank and variants)."
)
__copyright__ = "Copyright (C) 2023-2025 Tzu-Chi Yen"
__license__ = "LGPL-3.0-or-later"
__author__ = "Tzu-Chi Yen"
__author_email__ = "tzuchi.yen@colorado.edu"
__url__ = "https://github.com/junipertcy/regrank"


# Submodules that we lazily expose at the top-level
_submodules: tuple[str, ...] = ("datasets", "io", "optimize", "stats", "draw")


# Public API: submodules plus selected top-level helpers and metadata
__all__: list[str] = [
    # Metadata
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__author_email__",
    "__url__",
    "__license__",
    "__copyright__",
    # Submodules
    *_submodules,
    # Helpers
    "show_config",
]


def _resolve_version() -> str:
    """Resolve the package version dynamically."""
    # 1) If installed, get version from package metadata
    if importlib_metadata:
        try:
            return importlib_metadata.version("regrank")
        except importlib_metadata.PackageNotFoundError:
            # Not installed, fall through to checking pyproject.toml
            pass

    # 2) For local dev, try reading pyproject.toml
    if tomllib:
        pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                pyproject_data = tomllib.load(f)
            # Try PEP 621 (project.version) or Poetry (tool.poetry.version)
            version = pyproject_data.get("project", {}).get(
                "version"
            ) or pyproject_data.get("tool", {}).get("poetry", {}).get("version")
            if version:
                return str(version)

    # 3) Fallback
    return "0.0.0.dev0"


__version__ = _resolve_version()


def __getattr__(name: str) -> Any:
    """
    Lazily import submodules on first access, following PEP 562.
    """
    if name in _submodules:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Expose the public API when dir() is called.
    """
    return __all__


def _get_dependency_versions() -> dict[str, str]:
    """Safely get versions of key optional dependencies."""
    deps = ["numpy", "scipy", "pandas", "matplotlib", "networkx"]
    versions = {}
    for mod_name in deps:
        try:
            mod = importlib.import_module(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "N/A")
        except ImportError:
            versions[mod_name] = "not installed"
    return versions


def show_config() -> None:
    """Show regrank runtime, platform, and dependency information."""
    print("regrank configuration:")
    print(f"  regrank version: {__version__}")

    u = platform.uname()
    print("\nPlatform information:")
    print(
        f"  python:     {platform.python_version()} ({platform.python_implementation()})"
    )
    print(f"  system:     {u.system} ({u.release})")
    print(f"  machine:    {u.machine}")

    print("\nDependency versions:")
    dep_versions = _get_dependency_versions()
    if not dep_versions:
        print("  No optional dependencies found.")
    else:
        for name, version in dep_versions.items():
            print(f"  {name:<12}{version}")
