"""QoR (quality-of-results) report writer.

Produces the .rpt summary expected after a macro-placement step: design
stats, HPWL before/after, proxy-cost breakdown (wirelength / density /
congestion), overlaps resolved, bounds status, and runtime.
"""

import time
from pathlib import Path

import torch


def hpwl_um(placement, benchmark) -> float:
    """Half-perimeter wirelength over the benchmark's mapped nets, microns."""
    total = 0.0
    pos = placement
    for k, nodes in enumerate(benchmark.net_nodes):
        pts = pos[nodes]
        span = pts.max(dim=0).values - pts.min(dim=0).values
        w = float(benchmark.net_weights[k]) if benchmark.num_nets else 1.0
        total += w * float(span.sum())
    return total


def overlap_pairs(placement, benchmark) -> int:
    """Hard-macro overlap pair count (the evaluator's validity criterion)."""
    n = benchmark.num_hard_macros
    if n < 2:
        return 0
    p = placement[:n]
    s = benchmark.macro_sizes[:n]
    dx = (p[:, None, 0] - p[None, :, 0]).abs()
    dy = (p[:, None, 1] - p[None, :, 1]).abs()
    ov = (dx < (s[:, None, 0] + s[None, :, 0]) / 2 - 1e-6) & \
         (dy < (s[:, None, 1] + s[None, :, 1]) / 2 - 1e-6)
    return int(torch.triu(ov, diagonal=1).sum())


def write_report(
    out_path,
    result,
    placement,
    runtime_s: float,
    initial_costs=None,
    final_costs=None,
    valid=None,
    violations=None,
    inputs=None,
    outputs=None,
):
    """Write the QoR report. Cost dicts come from compute_proxy_cost."""
    b = result.benchmark
    initial = b.macro_positions
    lines = [
        "=" * 64,
        " Macro placement QoR report",
        f" design   : {result.design.name}",
        f" date     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f" placer   : varrahan v2 (eda_io flow)",
        "=" * 64,
        "",
        "[design]",
        f"  canvas            : {b.canvas_width:.2f} x {b.canvas_height:.2f} um",
        f"  hard macros       : {b.num_hard_macros}"
        f"  (fixed: {int(b.macro_fixed.sum())})",
        f"  soft macros       : {b.num_soft_macros}"
        f"  (std-cell clusters)",
        f"  std cells clustered: {sum(len(m) for m in result.soft_members)}",
        f"  i/o ports         : {len(result.port_names)}",
        f"  nets mapped       : {b.num_nets}",
    ]
    if result.dropped:
        lines.append(f"  components dropped: {len(result.dropped)}"
                     f" (no master geometry)")
    if inputs:
        lines += ["", "[inputs]"] + [f"  {k:<8}: {v}" for k, v in inputs.items()]
    if outputs:
        lines += ["", "[outputs]"] + [f"  {k:<8}: {v}" for k, v in outputs.items()]

    lines += [
        "",
        "[wirelength]",
        f"  HPWL initial      : {hpwl_um(initial, b):,.1f} um",
        f"  HPWL placed       : {hpwl_um(placement, b):,.1f} um",
        "",
        "[legality]",
        f"  hard overlaps initial : {overlap_pairs(initial, b)}",
        f"  hard overlaps placed  : {overlap_pairs(placement, b)}",
    ]
    if valid is not None:
        lines.append(f"  evaluator validity    : "
                     f"{'PASS' if valid else 'FAIL'}")
        if not valid and violations:
            for v in list(violations)[:5]:
                lines.append(f"    - {v}")

    def cost_block(label, costs):
        return [
            f"  {label:<8} proxy {costs['proxy_cost']:.4f}  "
            f"(wl {costs['wirelength_cost']:.4f}"
            f" | den {costs['density_cost']:.4f}"
            f" | cong {costs['congestion_cost']:.4f})",
        ]

    if initial_costs or final_costs:
        lines += ["", "[proxy cost]  (1.0*wl + 0.5*den + 0.5*cong, TILOS exact)"]
        if initial_costs:
            lines += cost_block("initial", initial_costs)
        if final_costs:
            lines += cost_block("placed", final_costs)
        if initial_costs and final_costs:
            d = float(final_costs["proxy_cost"]) - float(initial_costs["proxy_cost"])
            lines.append(f"  delta    {d:+.4f}")

    lines += [
        "",
        "[runtime]",
        f"  placer            : {runtime_s:.1f} s",
        "",
    ]
    Path(out_path).write_text("\n".join(lines))
