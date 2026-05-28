"""Why does relocation help some benchmarks far more than others (ibm04/ibm02 vs
ibm01/ibm09)? Compute per-benchmark STRUCTURAL features that should predict
relocation leverage, so we can correlate them with the measured R2 gains:

  - hard_util      : sum(hard-macro area) / canvas area  (open space to move INTO)
  - max_frac       : largest hard macro's area / total hard area  (dominant macro)
  - top5_frac      : top-5 hard macros' area / total hard area     (concentration)
  - n_hard         : hard-macro count
  - baseline cong  : weighted congestion of the legalized baseline (headroom)

Hypothesis: high gain ↔ few dominant macros (high max_frac/top5_frac) + open
space (low hard_util) + high baseline congestion (more to relieve).

    uv run python submissions/varrahan/v2/test/diagnostic/_reloc_leverage.py
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
V2_DIR = THIS.parents[2]
REPO_ROOT = THIS.parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR))

from macro_place.loader import load_benchmark_from_dir  # noqa: E402

_spec = importlib.util.spec_from_file_location("v2_placer", str(V2_DIR / "placer.py"))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_will_legalize = _v2._will_legalize
_exact_proxy = _v2._exact_proxy
_load_plc = _v2._load_plc

ICCAD_DIR = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"

# Committed R1 (1.4326) per-benchmark proxies, for gain = R1 - R2 overlay.
R1 = {
    "ibm01": 1.1107, "ibm02": 1.4759, "ibm03": 1.2297, "ibm04": 1.2349,
    "ibm06": 1.5354, "ibm07": 1.4733, "ibm08": 1.4859, "ibm09": 1.1005,
    "ibm10": 1.3231, "ibm11": 1.2166, "ibm12": 1.6309, "ibm13": 1.3673,
    "ibm14": 1.5706, "ibm15": 1.5826, "ibm16": 1.5009, "ibm17": 1.7320,
    "ibm18": 1.7837,
}


def _parse_r2(log_path):
    """Map ICCAD benchmark order → final proxy from an --all log."""
    order = ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08",
             "ibm09", "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15",
             "ibm16", "ibm17", "ibm18"]
    vals = []
    try:
        for line in Path(log_path).read_text().splitlines():
            s = line.strip()
            if s.startswith("proxy="):
                vals.append(float(s.split("proxy=")[1].split()[0]))
    except FileNotFoundError:
        return {}
    return dict(zip(order, vals))


def features(name):
    bm, _ = load_benchmark_from_dir(str(ICCAD_DIR / name))
    plc = _load_plc(name, bm)
    n = bm.num_hard_macros
    cw, ch = bm.canvas_width, bm.canvas_height
    sizes = bm.macro_sizes[:n].numpy().astype(np.float64)
    areas = sizes[:, 0] * sizes[:, 1]
    tot = float(areas.sum())
    hard_util = tot / (cw * ch)
    srt = np.sort(areas)[::-1]
    max_frac = float(srt[0] / tot) if tot > 0 else 0.0
    top5_frac = float(srt[:5].sum() / tot) if tot > 0 else 0.0

    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = (bm.get_movable_mask() & bm.get_hard_macro_mask())[:n].numpy()
    init = bm.macro_positions[:n].numpy().copy().astype(np.float64)
    leg = _will_legalize(init, movable, sizes, hw, hh, cw, ch, n)
    pl = bm.macro_positions.clone()
    pl[:n, 0] = torch.tensor(leg[:, 0], dtype=torch.float32)
    pl[:n, 1] = torch.tensor(leg[:, 1], dtype=torch.float32)
    _exact_proxy(pl, bm, plc)
    base_cong = 0.5 * float(plc.get_congestion_cost())
    return dict(n_hard=n, n_soft=bm.num_soft_macros, hard_util=hard_util,
                max_frac=max_frac, top5_frac=top5_frac, base_cong=base_cong)


def main():
    r2 = _parse_r2("/tmp/r2_all.log")
    names = list(R1.keys())
    rows = []
    for nm in names:
        try:
            f = features(nm)
            gain = R1[nm] - r2.get(nm, R1[nm])
            rows.append((nm, gain, f))
        except Exception as e:
            print(f"{nm}: ERROR {e}")
    rows.sort(key=lambda r: -r[1])  # biggest gain first
    print(f"\n{'bench':6} {'R2gain':>7} {'n_hard':>6} {'hard_util':>9} "
          f"{'max_frac':>8} {'top5_frac':>9} {'base_cong':>9}")
    for nm, gain, f in rows:
        print(f"{nm:6} {gain:>+7.4f} {f['n_hard']:>6} {f['hard_util']:>9.3f} "
              f"{f['max_frac']:>8.3f} {f['top5_frac']:>9.3f} {f['base_cong']:>9.3f}")


if __name__ == "__main__":
    main()
