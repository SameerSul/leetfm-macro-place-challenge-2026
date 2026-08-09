import base64
import json
from pathlib import Path

import numpy as np
import pytest

from dreamplace_bridge.run_bridge import _decode_progress_payload, _use_final_cache


def test_progress_payload_float32_round_trip():
    xy = np.asarray([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32)
    payload = {
        "iteration": 10,
        "count": 2,
        "dtype": "float32",
        "coordinates": base64.b64encode(xy.tobytes()).decode("ascii"),
    }
    decoded = np.frombuffer(
        base64.b64decode(json.dumps(payload) and payload["coordinates"]), dtype=np.float32
    )
    np.testing.assert_array_equal(decoded.reshape((-1, 2)), xy)
    iteration, decoded_record = _decode_progress_payload(
        "VIVAPLACE_PROGRESS " + json.dumps(payload)
    )
    assert iteration == 10
    np.testing.assert_array_equal(decoded_record, xy)


def test_malformed_progress_and_cache_bypass_isolation():
    with pytest.raises(ValueError):
        _decode_progress_payload(
            'VIVAPLACE_PROGRESS {"iteration":10,"count":2,"dtype":"float64","coordinates":""}'
        )
    assert _use_final_cache(None)
    assert not _use_final_cache(lambda _event: None)


def test_bootstrap_applies_tracked_progress_patch():
    root = Path(__file__).resolve().parents[2]
    bootstrap = (root / "scripts/dreamplace/bootstrap.sh").read_text()
    patcher = (root / "scripts/dreamplace/apply_visualizer_patch.py").read_text()
    assert "apply_visualizer_patch.py" in bootstrap
    assert "VIVAPLACE_PROGRESS_EVERY" in patcher
