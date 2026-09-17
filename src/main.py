import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placer.pipeline.macro_placer import MacroPlacer as _MacroPlacer


class MacroPlacer(_MacroPlacer):
    """Local class required by the evaluator's module-based discovery."""


__all__ = ["MacroPlacer"]
