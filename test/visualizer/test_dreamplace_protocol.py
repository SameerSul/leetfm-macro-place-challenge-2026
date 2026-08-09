import base64
import json
from pathlib import Path

import numpy as np
import pytest

from dreamplace_bridge.run_bridge import _decode_progress_payload, _use_final_cache
from scripts.dreamplace.apply_visualizer_patch import BLOCK_END, BLOCK_START, CALL_MARKER, patch


def test_progress_payload_float32_round_trip():
    xy = np.asarray([[1.25, 2.5], [3.75, 4.0]], dtype=np.float32)
    payload = {
        "protocol": 1,
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
    with pytest.raises(ValueError, match="unsupported DREAMPlace progress protocol"):
        _decode_progress_payload(
            'VIVAPLACE_PROGRESS {"protocol":2,"iteration":10,"count":0,'
            '"dtype":"float32","coordinates":""}'
        )
    assert _use_final_cache(None)
    assert not _use_final_cache(lambda _event: None)


def test_bootstrap_applies_tracked_progress_patch():
    root = Path(__file__).resolve().parents[2]
    bootstrap = (root / "scripts/dreamplace/bootstrap.sh").read_text()
    patcher = (root / "scripts/dreamplace/apply_visualizer_patch.py").read_text()
    assert "apply_visualizer_patch.py" in bootstrap
    assert "--check" in bootstrap
    assert "4c64c3f49eca86ccf5d5a050c92e030352cc8d62" in bootstrap
    assert "VIVAPLACE_PROGRESS_EVERY" in patcher


def test_tracked_patcher_supports_current_dreamplace_source(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / "dreamplace_src/dreamplace/NonLinearPlace.py"
    target = tmp_path / "NonLinearPlace.py"
    text = source.read_text()
    start = text.index(BLOCK_START)
    end = text.index(BLOCK_END, start) + len(BLOCK_END)
    text = text[:start] + text[end:]
    text = text.replace(CALL_MARKER, "", 1)
    text = text.replace("import base64\n", "", 1).replace("import json\n", "", 1)
    target.write_text(text)
    assert patch(target)
    assert not patch(target)
    patch(target, check=True)
    text = target.read_text()
    assert text.count(BLOCK_START) == 1
    assert text.count("emit_vivaplace_progress(model.data_collections.pos[0]") == 1
