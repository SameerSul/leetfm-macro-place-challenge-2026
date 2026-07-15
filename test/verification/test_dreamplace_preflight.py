import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dreamplace"))

from preflight import DEFAULT_BUILD_ROOT, probe


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(
    not (DEFAULT_BUILD_ROOT / "install" / "dreamplace" / "Placer.py").exists(),
    reason="run scripts/dreamplace/bootstrap.sh all to install DREAMPlace",
)
def test_dreamplace_native_extensions_match_pinned_python_abi():
    ok, detail = probe(DEFAULT_BUILD_ROOT)

    assert ok, detail
