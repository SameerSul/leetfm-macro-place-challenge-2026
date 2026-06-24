import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placer.pipeline.macro_placer import MacroPlacer as _MacroPlacer


class MacroPlacer(_MacroPlacer):
    def __init__(self, *args, **kwargs):
        # Used by data-collection sweeps; normal runs keep the default seed.
        env_seed = os.environ.get("SEED")
        if env_seed is not None and "seed" not in kwargs:
            kwargs["seed"] = int(env_seed)
        super().__init__(*args, **kwargs)


__all__ = ["MacroPlacer"]
