"""Make the Update 1 modules (common, lid, labels, dataset) importable from update2/.

Import this module before any Update 1 import:

    import setup_path  # noqa: F401
    from dataset import make_loader
"""
from __future__ import annotations

import sys
from pathlib import Path

UPDATE2_DIR = Path(__file__).resolve().parent
UPDATE1_DIR = UPDATE2_DIR.parent / "update1"

if str(UPDATE1_DIR) not in sys.path:
    # after the script's own directory, before site-packages
    sys.path.insert(1, str(UPDATE1_DIR))
