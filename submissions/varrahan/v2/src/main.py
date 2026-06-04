import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placer.pipeline import macro_placer as _impl
from placer.pipeline.macro_placer import MacroPlacer as _MacroPlacer


# The evaluator's loader (macro_place/evaluate.py:_load_placer) only accepts a
# placer class whose __module__ equals this file's stem ("main"). A bare
# re-import keeps __module__ == "placer.pipeline.macro_placer" and gets skipped,
# so define a local subclass that is owned by this module.
class MacroPlacer(_MacroPlacer):
    pass


__all__ = ["MacroPlacer"]

def __getattr__(name):
    try:
        return getattr(_impl, name)
    except AttributeError:
        import placer
        return getattr(placer, name)
