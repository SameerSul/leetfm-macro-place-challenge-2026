import importlib.util
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Default model for hard-relocation candidate filtering.
ML_FILTER_MANIFEST = (
    SRC_DIR.parent / "ml_data" / "models" / "clean-wide32-holdout-ibm13-001" / "manifest.json"
)


def _enable_ml_filter_defaults() -> None:
    """Use the shipped ranker unless the caller chose ML settings."""
    if any(key.startswith("ML_") for key in os.environ):
        return
    if not ML_FILTER_MANIFEST.is_file() or importlib.util.find_spec("xgboost") is None:
        return
    os.environ["ML_MODEL_MANIFEST"] = str(ML_FILTER_MANIFEST)
    os.environ["ML_FILTER_OPERATORS"] = "hard_relocation"
    os.environ["ML_FILTER_TOP_K"] = "16"
    os.environ["ML_HARD_RELOCATION_N_TARGETS"] = "32"


_enable_ml_filter_defaults()

from placer.pipeline import macro_placer as _impl
from placer.pipeline.macro_placer import MacroPlacer as _MacroPlacer


class MacroPlacer(_MacroPlacer):
    def __init__(self, *args, **kwargs):
        # Used by data-collection sweeps; normal runs keep the default seed.
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
