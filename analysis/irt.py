"""Compatibility layer for IRT fitting and figure helpers.

New code should import model/table APIs from ``analysis.irt_model`` and plotting
APIs from ``analysis.irt_figures``. This module keeps existing imports working.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.irt_model import *  # noqa: F401,F403
from analysis.irt_figures import *  # noqa: F401,F403
from analysis.irt_model import main


if __name__ == "__main__":
    raise SystemExit(main())
