"""The single source of truth for the package version.

Releasing is manual: bump this string, merge, then dispatch ``publish.yml``. It lives in its own
module so ``_client.py`` can read it for the ``user-agent`` header without importing the package
root, which would be a cycle.
"""

from __future__ import annotations

__version__ = "0.1.0"
