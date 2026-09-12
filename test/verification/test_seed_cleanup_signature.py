"""Keep the injected seed cleanup call compatible with the real relocation API."""

import ast
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from placer.local_search.relocation import _soft_relocation_moves
from placer.pipeline.segments.floorplan_seed import run_seed_portfolio


def test_seed_soft_cleanup_uses_current_relocation_signature():
    tree = ast.parse(inspect.getsource(run_seed_portfolio))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "soft_relocation_fn"
    ]
    assert len(calls) == 1
    call = calls[0]
    values = [arg.id for arg in call.args]
    bound = inspect.signature(_soft_relocation_moves).bind(
        *values,
        **{kw.arg: None for kw in call.keywords},
    )
    assert bound.arguments["benchmark"] == "benchmark"
    assert bound.arguments["incremental_scorer"] == "cand_scorer"
    assert bound.arguments["initial_score"] == "score"
