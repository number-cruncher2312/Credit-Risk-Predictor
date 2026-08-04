"""Pytest bootstrap for importing top-level project modules.

This keeps `from reg_capital import ...` working even when pytest is launched
from a context that does not automatically place the repository root on
`sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)
