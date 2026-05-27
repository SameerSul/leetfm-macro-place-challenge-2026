"""
Compare v2's vectorized _exact_proxy vs the framework's reference compute_proxy_cost
on the same placement, to isolate where the ~0.03 scoring discrepancy on ibm03 comes from.

Usage:
    uv run python submissions/varrahan/v2/tests/diagnostic/_scoring_diagnostic.py ibm03
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

V2_PATH = Path(__file__).resolve().parents[2] / "placer.py"


def _load_v2_module():
    spec = importlib.util.spec_from_file_location("v2_placer_mod", V2_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_vectorized(placement, benchmark, plc, mod):
    """Call v2's _exact_proxy and return components."""
    mod._patch_plc_wirelength(plc)
    mod._patch_plc_congestion(plc, benchmark)
    placement_np = placement.cpu().numpy()
    mod._fast_set_placement(plc, placement_np, benchmark)
    wl = plc.get_cost()
    dens = plc.get_density_cost()
    cong = plc.get_congestion_cost()
    return {
        "wl": float(wl),
        "den": float(dens),
        "cong": float(cong),
        "proxy": float(wl + 0.5 * dens + 0.5 * cong),
    }


def score_reference(placement, benchmark, plc):
    """Call the framework's compute_proxy_cost and return components."""
    r = compute_proxy_cost(placement, benchmark, plc)
    return {
        "wl": float(r["wirelength_cost"]),
        "den": float(r["density_cost"]),
        "cong": float(r["congestion_cost"]),
        "proxy": float(r["proxy_cost"]),
    }


def fmt_components(label, r):
    return (
        f"{label}: proxy={r['proxy']:.4f}  wl={r['wl']:.4f}  "
        f"den={r['den']:.4f}  cong={r['cong']:.4f}"
    )


def diff_components(a, b, label_a, label_b):
    print(f"\nDelta ({label_a} - {label_b}):")
    print(f"  proxy: {a['proxy'] - b['proxy']:+.6f}")
    print(f"  wl   : {a['wl']   - b['wl']  :+.6f}")
    print(f"  den  : {a['den']  - b['den'] :+.6f}")
    print(f"  cong : {a['cong'] - b['cong']:+.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", default="ibm03", nargs="?")
    args = ap.parse_args()

    mod = _load_v2_module()

    bdir = str(ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.benchmark)
    print(f"Loading benchmark {args.benchmark} from {bdir}...")
    benchmark, plc_pristine = load_benchmark_from_dir(bdir)
    _, plc_vec = load_benchmark_from_dir(bdir)
    placement = benchmark.macro_positions.clone()  # initial placement from .plc
    print(f"Placement shape: {tuple(placement.shape)}")

    # --- score 1: pure reference on pristine plc ---
    pristine_first = score_reference(placement, benchmark, plc_pristine)
    print("\n" + fmt_components("PRISTINE (reference, fresh plc)    ", pristine_first))

    # --- score 2: v2 vectorized _exact_proxy ---
    vec = score_vectorized(placement, benchmark, plc_vec, mod)
    print(fmt_components("VECTOR   (v2 _exact_proxy)         ", vec))

    # --- score 3: framework re-score on the now-patched plc ---
    #   This is what the harness sees: the placer hands back a plc it has
    #   patched, then the harness calls compute_proxy_cost on it.
    framework_repeat = score_reference(placement, benchmark, plc_vec)
    print(fmt_components("FRAMEWORK (post-vec, patched plc)  ", framework_repeat))

    diff_components(vec, pristine_first, "VECTOR", "PRISTINE")
    diff_components(framework_repeat, pristine_first, "FRAMEWORK_PATCHED", "PRISTINE")
    diff_components(framework_repeat, vec, "FRAMEWORK_PATCHED", "VECTOR")

    # Re-score on the pristine plc AGAIN to confirm stability
    pristine_again = score_reference(placement, benchmark, plc_pristine)
    diff_components(pristine_again, pristine_first, "PRISTINE-2nd", "PRISTINE-1st")

    # -----------------------------------------------------------------------
    # Stateful test: perturb, score via vectorized path several times to
    # exercise the cache, then score via reference. If they disagree at the
    # end on the SAME final placement, the bug is in stateful interaction
    # between _fast_set_placement and the reference _set_placement path.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STATEFUL TEST: perturb-and-restore")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n = placement.shape[0]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

    # Make several perturbed placements and score each via the vectorized
    # path on plc_vec. This builds up cache state.
    base_np = placement.cpu().numpy().copy()
    for it in range(5):
        pert = base_np.copy()
        # only move HARD macros (soft are fixed in baseline)
        n_hard = benchmark.num_hard_macros
        delta = rng.uniform(-0.05, 0.05, size=(n_hard, 2))
        pert[:n_hard, 0] += delta[:, 0] * cw
        pert[:n_hard, 1] += delta[:, 1] * ch
        # Honor canvas bounds
        pert[:n_hard, 0] = np.clip(pert[:n_hard, 0], 0, cw)
        pert[:n_hard, 1] = np.clip(pert[:n_hard, 1], 0, ch)
        import torch as _torch
        pert_t = _torch.from_numpy(pert.astype(np.float32))
        r_vec = score_vectorized(pert_t, benchmark, plc_vec, mod)
        print(f"  iter {it}: vec proxy={r_vec['proxy']:.4f}")

    # Now restore to the original placement on plc_vec via the vectorized path
    print("\nNow restore to the ORIGINAL placement and compare paths:")
    r_vec_restored = score_vectorized(placement, benchmark, plc_vec, mod)
    print(fmt_components("VECTOR-restored (after 5 perturbs) ", r_vec_restored))

    # Re-score same placement via reference path on the same plc_vec
    r_ref_after_vec = score_reference(placement, benchmark, plc_vec)
    print(fmt_components("REFERENCE on plc_vec after vec     ", r_ref_after_vec))

    # And score on the pristine plc (which never saw perturbations)
    r_pristine_again = score_reference(placement, benchmark, plc_pristine)
    print(fmt_components("PRISTINE (unchanged)               ", r_pristine_again))

    diff_components(r_vec_restored, r_pristine_again, "VEC-restored", "PRISTINE")
    diff_components(r_ref_after_vec, r_pristine_again, "REF-after-vec", "PRISTINE")
    diff_components(r_vec_restored, r_ref_after_vec, "VEC-restored", "REF-after-vec")


if __name__ == "__main__":
    main()
