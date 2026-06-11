"""Verify the shipped hard-relocation ranker is connected by default.

Importing src/main.py with no ML_* env vars must enable the S10 config-B
filter (wide-32 pool, ranker keeps 16), and any preset ML_* var must keep
the defaults out of the way.

Run: uv run python submissions/varrahan/v2/test/verification/_verify_ml_filter_wiring.py
"""

import os
import sys
from pathlib import Path

V2_SRC = Path(__file__).resolve().parents[2] / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

# Case 1: clean environment -> importing main connects the ranker.
for key in [k for k in os.environ if k.startswith("ML_")]:
    del os.environ[key]

import main  # noqa: E402  (import runs _enable_ml_filter_defaults)

assert main.ML_FILTER_MANIFEST.is_file(), f"missing model manifest: {main.ML_FILTER_MANIFEST}"
assert os.environ.get("ML_MODEL_MANIFEST") == str(main.ML_FILTER_MANIFEST)
assert os.environ.get("ML_FILTER_OPERATORS") == "hard_relocation"
assert os.environ.get("ML_FILTER_TOP_K") == "16"
assert os.environ.get("ML_HARD_RELOCATION_N_TARGETS") == "32"

from placer.ml.shadow import (  # noqa: E402
    filter_candidate_indices,
    get_shadow_model_bank,
    is_filter_enabled,
)

assert is_filter_enabled("hard_relocation")
bank = get_shadow_model_bank()
assert bank is not None, "model bank failed to load"
assert bank.get("hard_relocation") is not None, "no hard_relocation ranker in bank"

# A synthetic wide-32 group must come back filtered to exactly 16 indices.
candidates = [
    {
        "operator": "hard_relocation",
        "candidate_rank": rank,
        "features": {
            "accepted_in_pass": 0,
            "source_hot_rank_norm": 0.1,
            "target_cold_rank_norm": rank / 31.0,
            "dx_norm": 0.01 * rank,
            "dy_norm": -0.01 * rank,
            "source_field_norm": 0.9,
            "target_field_norm": 0.02 * rank,
        },
    }
    for rank in range(32)
]
selected = filter_candidate_indices(operator="hard_relocation", candidates=candidates)
assert len(selected) == 16, f"expected 16 of 32 selected, got {len(selected)}"
assert all(0 <= idx < 32 for idx in selected)

# Case 2: any preset ML_* var (here an explicit disable) skips the defaults.
for key in [k for k in os.environ if k.startswith("ML_")]:
    del os.environ[key]
os.environ["ML_FILTER_OPERATORS"] = ""
main._enable_ml_filter_defaults()
assert "ML_MODEL_MANIFEST" not in os.environ, "defaults must not fire when ML_* vars are set"
assert not is_filter_enabled("hard_relocation")

print(f"ML filter wiring verified: manifest={main.ML_FILTER_MANIFEST.parent.name}, "
      f"filtered 32 -> {len(selected)} candidates, opt-out respected")
