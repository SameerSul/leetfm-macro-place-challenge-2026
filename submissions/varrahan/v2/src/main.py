import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placer.pipeline import macro_placer as _impl
from placer.pipeline.macro_placer import MacroPlacer as _MacroPlacer


class MacroPlacer(_MacroPlacer):
    def __init__(self, *args, **kwargs):
        # V2_SEED lets the ML collection script sweep seeds (the evaluator
        # constructs the placer with no arguments, so the seed is otherwise pinned
        # at the base default). Unset in real evaluation, so behavior is unchanged.
        env_seed = os.environ.get("V2_SEED")
        if env_seed is not None and "seed" not in kwargs:
            kwargs["seed"] = int(env_seed)
        super().__init__(*args, **kwargs)


__all__ = ["MacroPlacer"]

def __getattr__(name):
    try:
        return getattr(_impl, name)
    except AttributeError:
        import placer
        return getattr(placer, name)
