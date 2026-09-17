"""Check constant-only placement settings and the evaluator entrypoint."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils import constants


def test_legacy_environment_cannot_override_seed_or_visualizer_cache(monkeypatch):
    from main import MacroPlacer
    from macro_place.evaluate import _load_placer
    from dreamplace_bridge.run_bridge import _use_final_cache

    monkeypatch.setenv("SEED", "invalid")
    monkeypatch.setenv("HIER_VISUALIZER_USE_CACHE", "1")
    monkeypatch.setattr(constants, "HIER_VISUALIZER_USE_CACHE", False)
    assert MacroPlacer().seed == 42
    assert MacroPlacer(seed=17).seed == 17
    assert _load_placer(ROOT / "src/main.py").seed == 42
    assert not _use_final_cache(lambda event: None)


def test_placement_and_diagnostics_have_no_environment_configuration():
    paths = [*(ROOT / "src").rglob("*.py"), *(ROOT / "test/diagnostic").glob("*.py")]
    for path in paths:
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                assert not {item.name for item in node.names} & {
                    "environ",
                    "getenv",
                    "putenv",
                    "unsetenv",
                }, path
            if not isinstance(node, ast.Attribute):
                continue
            assert node.attr not in {"getenv", "putenv", "unsetenv"}, path
            if node.attr != "environ":
                continue
            parent = parents[node]
            # Native subprocess setup and cuBLAS determinism have no tuning overrides.
            native_copy = (
                path == ROOT / "src/dreamplace_bridge/run_bridge.py"
                and isinstance(parent, ast.Attribute)
                and parent.attr == "copy"
            )
            cublas_setup = (
                path == ROOT / "test/diagnostic/replay_gpu_soft_refinement.py"
                and isinstance(parent, ast.Subscript)
                and isinstance(parent.ctx, ast.Store)
                and isinstance(parent.slice, ast.Constant)
                and parent.slice.value == "CUBLAS_WORKSPACE_CONFIG"
            )
            assert native_copy or cublas_setup, (path, node.lineno)
