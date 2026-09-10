"""Reject a requested budget override when the selected placer cannot use it."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test.benchmarks import run_synthetic


def test_unsupported_budget_is_reported_before_loading_design(monkeypatch, tmp_path, capsys):
    case = tmp_path / "testcases" / "example"
    case.mkdir(parents=True)
    (case / "netlist.pb.txt").touch()
    monkeypatch.setattr(run_synthetic, "OUT", tmp_path)
    monkeypatch.setattr(run_synthetic, "_load_placer", lambda path: object())
    monkeypatch.setattr(sys, "argv", ["run_synthetic.py", "--budget", "12"])

    with pytest.raises(SystemExit) as error:
        run_synthetic.main()

    assert error.value.code == 2
    assert "selected placer does not support --budget" in capsys.readouterr().err
