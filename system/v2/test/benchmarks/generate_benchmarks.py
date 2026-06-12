"""Synthetic benchmark generator for anti-overfitting evaluation.

The 17 IBM ICCAD04 benchmarks are homogeneous: square canvases, zero fixed
macros, identical routing capacities (66/107), ~0.8 total utilization,
lognormal-ish small macros, and a hand-tuned spread seed. A placer tuned on
them can silently overfit to those properties. This script generates testcases
that vary exactly those axes while staying in the same ICCAD04 protobuf format
(netlist.pb.txt + initial.plc), so they load and score through the unmodified
TILOS PlacementCost evaluator.

Outputs (under this directory):
    testcases/<name>/netlist.pb.txt   protobuf netlist (CT format)
    testcases/<name>/initial.plc      seed placement + canvas/grid/routing meta
    processed/<name>.pt               Benchmark tensor mirror (fast loading)
    metadata/<name>.json              generation config + axis description

Usage:
    uv run python system/v2/test/benchmarks/generate_benchmarks.py
    uv run python .../generate_benchmarks.py --only syn02_fixed
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Suite configuration ──────────────────────────────────────────────────────


@dataclass
class Cfg:
    name: str
    axis: str
    canvas_w: float
    canvas_h: float
    grid_cols: int
    grid_rows: int
    n_hard: int
    n_soft: int
    hard_util: float
    soft_util: float
    n_nets: int
    n_clusters: int = 6
    p_intra: float = 0.80
    n_ports: int = 160
    port_side_weights: tuple = (0.25, 0.25, 0.25, 0.25)  # LEFT, RIGHT, TOP, BOTTOM
    hard_shapes: tuple = ()  # explicit (w, h) shapes; empty = lognormal sizes
    n_fixed: int = 0
    routes_h: float = 66.0
    routes_v: float = 107.0
    macro_routes_h: float = 30.3
    macro_routes_v: float = 71.3
    seed_style: str = "spread"  # "spread" (shelf-packed, IBM-like) or "random"
    port_net_frac: float = 0.08
    net_sigma_frac: float = 0.045  # sink-locality radius as fraction of min canvas dim
    rng_seed: int = 0


SUITE = [
    Cfg(
        name="syn01_wide",
        axis="Non-square canvas (2.6:1 aspect). IBM canvases are all square.",
        canvas_w=65.0, canvas_h=25.0, grid_cols=62, grid_rows=24,
        n_hard=280, n_soft=1000, hard_util=0.40, soft_util=0.36, n_nets=14000,
        rng_seed=101,
    ),
    Cfg(
        name="syn02_fixed",
        axis="Pre-placed fixed macros (corners + center). IBM has zero fixed "
             "macros, so fixed-macro handling is never exercised.",
        canvas_w=42.0, canvas_h=42.0, grid_cols=42, grid_rows=42,
        n_hard=266, n_soft=950, hard_util=0.45, soft_util=0.30, n_nets=13000,
        n_fixed=6, n_clusters=5, rng_seed=102,
    ),
    Cfg(
        name="syn03_sram",
        axis="Commercial-style: few large uniform SRAM macros + low routing "
             "capacity (bp_quad-like 12.5/13.5). IBM has hundreds of small "
             "macros and fixed 66/107 capacity.",
        canvas_w=55.0, canvas_h=55.0, grid_cols=36, grid_rows=36,
        n_hard=40, n_soft=550, hard_util=0.42, soft_util=0.26, n_nets=16000,
        hard_shapes=((1.0, 0.62), (0.78, 0.78), (1.25, 0.5)),
        n_clusters=4, p_intra=0.75, n_ports=120,
        routes_h=12.5, routes_v=13.5, macro_routes_h=5.5, macro_routes_v=6.5,
        rng_seed=103,
    ),
    Cfg(
        name="syn04_dense",
        axis="High utilization (~0.90 total): little whitespace for "
             "legalization. IBM sits near 0.80.",
        canvas_w=38.0, canvas_h=38.0, grid_cols=40, grid_rows=40,
        n_hard=300, n_soft=1100, hard_util=0.52, soft_util=0.38, n_nets=15000,
        rng_seed=104,
    ),
    Cfg(
        name="syn05_sparse",
        axis="Low utilization (~0.33 total): congestion driven by net "
             "topology, not packing pressure.",
        canvas_w=62.0, canvas_h=62.0, grid_cols=40, grid_rows=40,
        n_hard=190, n_soft=700, hard_util=0.17, soft_util=0.16, n_nets=11000,
        n_clusters=5, n_ports=140, rng_seed=105,
    ),
    Cfg(
        name="syn06_cluster",
        axis="Strongly clustered Rent-style netlist (9 communities, 92% "
             "intra-cluster nets): rewards community-aware placement, "
             "punishes blind spreading.",
        canvas_w=45.0, canvas_h=45.0, grid_cols=43, grid_rows=43,
        n_hard=320, n_soft=1200, hard_util=0.38, soft_util=0.36, n_nets=16000,
        n_clusters=9, p_intra=0.92, rng_seed=106,
    ),
    Cfg(
        name="syn07_ports",
        axis="I/O-heavy with 70% of ports on the LEFT edge: creates a "
             "directional wirelength/congestion pull absent from IBM.",
        canvas_w=40.0, canvas_h=40.0, grid_cols=41, grid_rows=41,
        n_hard=250, n_soft=900, hard_util=0.40, soft_util=0.34, n_nets=13000,
        n_ports=320, port_side_weights=(0.70, 0.10, 0.10, 0.10),
        port_net_frac=0.25, rng_seed=107,
    ),
    Cfg(
        name="syn08_routes",
        axis="Inverted routing capacity (H=107 > V=66, macro routes swapped). "
             "All IBM benchmarks share V-dominant 66/107.",
        canvas_w=45.0, canvas_h=45.0, grid_cols=43, grid_rows=43,
        n_hard=280, n_soft=1000, hard_util=0.40, soft_util=0.36, n_nets=14000,
        routes_h=107.0, routes_v=66.0, macro_routes_h=71.3, macro_routes_v=30.3,
        rng_seed=108,
    ),
    Cfg(
        name="syn09_seedless",
        axis="Random initial placement: no hand-tuned spread seed to lean on "
             "(IBM initial.plc comes from a prior EDA flow).",
        canvas_w=45.0, canvas_h=45.0, grid_cols=43, grid_rows=43,
        n_hard=280, n_soft=1000, hard_util=0.40, soft_util=0.36, n_nets=14000,
        seed_style="random", rng_seed=109,
    ),
    Cfg(
        name="syn10_xl",
        axis="Scale stress: ~820 hard / 2000 soft macros, 50x50 grid, 26k "
             "nets - beyond ibm17 in macro count and grid cells.",
        canvas_w=80.0, canvas_h=80.0, grid_cols=50, grid_rows=50,
        n_hard=820, n_soft=2000, hard_util=0.45, soft_util=0.28, n_nets=26000,
        n_clusters=12, n_ports=300, rng_seed=110,
    ),
]


# ── Geometry helpers ─────────────────────────────────────────────────────────


def split_regions(x0, y0, w, h, k):
    """Slice-and-dice the canvas into k roughly equal rectangular regions."""
    if k == 1:
        return [(x0, y0, w, h)]
    k1 = k // 2
    frac = k1 / k
    if w >= h:
        return split_regions(x0, y0, w * frac, h, k1) + split_regions(
            x0 + w * frac, y0, w * (1 - frac), h, k - k1
        )
    return split_regions(x0, y0, w, h * frac, k1) + split_regions(
        x0, y0 + h * frac, w, h * (1 - frac), k - k1
    )


def sample_hard_sizes(cfg, rng):
    """Sample hard macro (w, h) scaled so total area hits hard_util."""
    canvas_area = cfg.canvas_w * cfg.canvas_h
    target = cfg.hard_util * canvas_area
    if cfg.hard_shapes:
        shapes = np.array([cfg.hard_shapes[i % len(cfg.hard_shapes)] for i in range(cfg.n_hard)])
        scale = np.sqrt(target / (shapes[:, 0] * shapes[:, 1]).sum())
        return shapes * scale
    areas = rng.lognormal(0.0, 0.9, cfg.n_hard)
    areas *= target / areas.sum()
    aspect = np.clip(np.exp(rng.normal(0.0, 0.35, cfg.n_hard)), 0.4, 2.5)
    w = np.sqrt(areas * aspect)
    h = np.sqrt(areas / aspect)
    max_dim = 0.16 * min(cfg.canvas_w, cfg.canvas_h)
    sizes = np.clip(np.stack([w, h], axis=1), 0.25, max_dim)
    # one rescale pass to restore the utilization target after clipping
    sizes *= np.sqrt(target / (sizes[:, 0] * sizes[:, 1]).sum())
    return np.clip(sizes, 0.2, max_dim * 1.2)


def sample_soft_sizes(cfg, rng):
    canvas_area = cfg.canvas_w * cfg.canvas_h
    target = cfg.soft_util * canvas_area
    areas = rng.lognormal(0.0, 0.7, cfg.n_soft)
    areas *= target / areas.sum()
    aspect = np.clip(np.exp(rng.normal(0.0, 0.3, cfg.n_soft)), 0.5, 2.0)
    w = np.sqrt(areas * aspect)
    h = np.sqrt(areas / aspect)
    return np.stack([w, h], axis=1)


def shelf_pack(rng, idx, sizes, region):
    """Shelf-pack macros into region with even spreading + jitter.

    Returns {macro_index: (cx, cy)}. Macros that don't fit fall back to a
    random in-region position (seed overlaps are allowed, like IBM seeds).
    """
    x0, y0, W, H = region
    order = sorted(idx, key=lambda i: -sizes[i][1])
    shelves, cur, cur_w, used_h = [], [], 0.0, 0.0
    leftover = []
    for i in order:
        w, h = sizes[i]
        if w > W or used_h + h > H:
            if cur and used_h + sizes[cur[0]][1] <= H:
                pass  # current shelf still open; macro itself too tall
            leftover.append(i)
            continue
        if cur_w + w <= W:
            cur.append(i)
            cur_w += w
        else:
            shelves.append(cur)
            used_h += sizes[cur[0]][1]
            if used_h + h > H:
                leftover.append(i)
                cur, cur_w = [], 0.0
                continue
            cur, cur_w = [i], w
    if cur:
        shelves.append(cur)
        used_h += sizes[cur[0]][1]

    pos = {}
    n_sh = len(shelves)
    gap_v = max(0.0, (H - used_h)) / (n_sh + 1) if n_sh else 0.0
    y = y0
    for shelf in shelves:
        y += gap_v
        shelf_h = sizes[shelf[0]][1]
        row_w = sum(sizes[i][0] for i in shelf)
        gap_h = max(0.0, (W - row_w)) / (len(shelf) + 1)
        x = x0
        for i in shelf:
            w, h = sizes[i]
            x += gap_h
            jx = rng.uniform(-0.4, 0.4) * gap_h
            jy = rng.uniform(-0.3, 0.3) * gap_v
            cx = np.clip(x + w / 2 + jx, x0 + w / 2, x0 + W - w / 2)
            cy = np.clip(y + h / 2 + jy, y0 + h / 2, y0 + H - h / 2)
            pos[i] = (cx, cy)
            x += w
        y += shelf_h
    for i in leftover:
        pos[i] = random_pos(rng, sizes[i], region)
    return pos


def random_pos(rng, size, region):
    x0, y0, W, H = region
    w, h = size
    cx = rng.uniform(x0 + w / 2, max(x0 + w / 2, x0 + W - w / 2))
    cy = rng.uniform(y0 + h / 2, max(y0 + h / 2, y0 + H - h / 2))
    return (cx, cy)


def place_fixed(cfg, sizes, fixed_ids):
    """Pin fixed macros at corners then along the canvas center line."""
    W, H = cfg.canvas_w, cfg.canvas_h
    spots = []
    for fx, fy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        spots.append((fx * W, fy * H))
    n_mid = max(0, len(fixed_ids) - 4)
    for j in range(n_mid):
        spots.append(((j + 1) / (n_mid + 1) * W, 0.5 * H))
    pos = {}
    for i, (cx, cy) in zip(fixed_ids, spots):
        w, h = sizes[i]
        pos[i] = (
            float(np.clip(cx, w / 2 + 0.1, W - w / 2 - 0.1)),
            float(np.clip(cy, h / 2 + 0.1, H - h / 2 - 0.1)),
        )
    return pos


# ── Netlist construction ─────────────────────────────────────────────────────


def gen_ports(cfg, rng):
    """Port positions on the die boundary; returns (positions, sides)."""
    sides = rng.choice(4, size=cfg.n_ports, p=np.array(cfg.port_side_weights))
    eps = 0.005
    pos, side_names = [], []
    for s in sides:
        if s == 0:  # LEFT
            pos.append((eps, rng.uniform(0.5, cfg.canvas_h - 0.5)))
            side_names.append("LEFT")
        elif s == 1:  # RIGHT
            pos.append((cfg.canvas_w - eps, rng.uniform(0.5, cfg.canvas_h - 0.5)))
            side_names.append("RIGHT")
        elif s == 2:  # TOP
            pos.append((rng.uniform(0.5, cfg.canvas_w - 0.5), cfg.canvas_h - eps))
            side_names.append("TOP")
        else:  # BOTTOM
            pos.append((rng.uniform(0.5, cfg.canvas_w - 0.5), eps))
            side_names.append("BOTTOM")
    return np.array(pos), side_names


def make_hard_pins(rng, sizes):
    """Pre-create pin pools per hard macro: (offsets, n_sink_pins)."""
    mean_area = (sizes[:, 0] * sizes[:, 1]).mean()
    pools = []
    for w, h in sizes:
        n_pins = int(np.clip(2 + 3 * np.sqrt((w * h) / mean_area), 2, 16))
        ox = rng.uniform(-0.45, 0.45, n_pins) * w
        oy = rng.uniform(-0.45, 0.45, n_pins) * h
        n_sink = max(1, int(0.6 * n_pins))
        pools.append({"off": np.stack([ox, oy], axis=1), "n_sink": n_sink, "next_drv": n_sink})
    return pools


def sample_degree(rng):
    r = rng.random()
    if r < 0.45:
        return 2
    if r < 0.82:
        return int(rng.integers(3, 6))
    if r < 0.96:
        return int(rng.integers(6, 13))
    return int(rng.integers(13, 25))


def build_nets(cfg, rng, hard_cluster, soft_cluster, port_cluster, hard_pins,
               hard_xy, soft_xy, port_xy):
    """Generate nets as driver->sinks endpoint name lists.

    Sink choice is spatially local (gaussian falloff around the driver) with a
    cluster preference, mimicking Rent-style locality of real netlists.

    Returns (port_inputs, hard_drv_inputs, soft_outputs):
        port_inputs:     {port_idx: [sink names]}
        hard_drv_inputs: {(macro_idx, pin_idx): [sink names]}
        soft_outputs:    {macro_idx: [[sink names], ...]}
    """
    n_clusters = cfg.n_clusters
    hard_by_c = [np.where(hard_cluster == c)[0] for c in range(n_clusters)]
    soft_by_c = [np.where(soft_cluster == c)[0] for c in range(n_clusters)]
    port_by_c = [np.where(port_cluster == c)[0] for c in range(n_clusters)]
    all_hard = np.arange(cfg.n_hard)
    all_soft = np.arange(cfg.n_soft)
    all_ports = np.arange(cfg.n_ports)
    sigma = cfg.net_sigma_frac * min(cfg.canvas_w, cfg.canvas_h)

    port_inputs, hard_drv_inputs, soft_outputs = {}, {}, {}
    free_ports = list(rng.permutation(cfg.n_ports)) if cfg.n_ports else []

    def local_choice(ids, xy, driver_xy, s):
        d2 = ((xy[ids] - driver_xy) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2 * s * s)) + 1e-12
        return int(rng.choice(ids, p=w / w.sum()))

    def pick_sink(cluster, used_macros, driver_xy):
        for _ in range(8):
            # intra-cluster sinks are tightly local; inter-cluster sinks are
            # still distance-weighted (mid-range), not uniform across the die
            if rng.random() < cfg.p_intra:
                h_ids, s_ids, p_ids = hard_by_c[cluster], soft_by_c[cluster], port_by_c[cluster]
                s_loc = sigma
            else:
                h_ids, s_ids, p_ids = all_hard, all_soft, all_ports
                s_loc = 3.0 * sigma
            r = rng.random()
            if r < 0.05 and len(p_ids):
                p = local_choice(p_ids, port_xy, driver_xy, s_loc)
                key = ("p", p)
                if key not in used_macros:
                    used_macros.add(key)
                    return f"p{p}"
            elif r < 0.35 and len(h_ids):
                m = local_choice(h_ids, hard_xy, driver_xy, s_loc)
                key = ("h", m)
                if key not in used_macros:
                    used_macros.add(key)
                    k = int(rng.integers(hard_pins[m]["n_sink"]))
                    return f"a{m}/IP{k}"
            elif len(s_ids):
                m = local_choice(s_ids, soft_xy, driver_xy, s_loc)
                key = ("s", m)
                if key not in used_macros:
                    used_macros.add(key)
                    return f"Grp_{m}/Pinput"
        return None

    for _ in range(cfg.n_nets):
        r = rng.random()
        if r < cfg.port_net_frac and free_ports:
            p = free_ports.pop()
            cluster = int(port_cluster[p])
            used = {("p", p)}
            driver = ("port", p)
            driver_xy = port_xy[p]
        elif r < cfg.port_net_frac + 0.62 or cfg.n_hard == 0:
            m = int(rng.integers(cfg.n_soft))
            cluster = int(soft_cluster[m])
            used = {("s", m)}
            driver = ("soft", m)
            driver_xy = soft_xy[m]
        else:
            m = int(rng.integers(cfg.n_hard))
            cluster = int(hard_cluster[m])
            used = {("h", m)}
            driver = ("hard", m)
            driver_xy = hard_xy[m]

        sinks = []
        for _ in range(sample_degree(rng) - 1):
            s = pick_sink(cluster, used, driver_xy)
            if s is not None:
                sinks.append(s)
        if not sinks:
            continue

        kind, m = driver
        if kind == "port":
            port_inputs[m] = sinks
        elif kind == "soft":
            soft_outputs.setdefault(m, []).append(sinks)
        else:
            pool = hard_pins[m]
            k = pool["next_drv"]
            if k >= len(pool["off"]):
                w_h = pool["off"][: pool["n_sink"]]  # reuse spread of sink offsets
                extra = w_h[int(rng.integers(len(w_h)))] * rng.uniform(0.5, 1.0)
                pool["off"] = np.vstack([pool["off"], extra])
            pool["next_drv"] = k + 1
            hard_drv_inputs[(m, k)] = sinks
    return port_inputs, hard_drv_inputs, soft_outputs


# ── Protobuf / plc writers ───────────────────────────────────────────────────


def _node(name, inputs, attrs):
    """Format one node block (parser is strict-positional: name, inputs, attrs)."""
    lines = ["node {", f'  name: "{name}"']
    for s in inputs or []:
        lines.append(f'  input: "{s}"')
    for key, kind, val in attrs:
        v = f"{val:.4f}" if kind == "f" else f'"{val}"'
        lines += [
            "  attr {",
            f'    key: "{key}"',
            "    value {",
            f"      {'f' if kind == 'f' else 'placeholder'}: {v}",
            "    }",
            "  }",
        ]
    lines.append("}")
    return "\n".join(lines)


def write_netlist(path, cfg, hard_sizes, hard_pos, fixed_mask, soft_sizes,
                  soft_pos, port_pos, port_sides, hard_pins,
                  port_inputs, hard_drv_inputs, soft_outputs):
    blocks = [
        _node("__metadata__", None, [("soft_macro_area_bloating_ratio", "f", 1.0)])
    ]
    for p in range(cfg.n_ports):
        x, y = port_pos[p]
        blocks.append(
            _node(
                f"p{p}",
                port_inputs.get(p),
                [("side", "placeholder", port_sides[p]),
                 ("type", "placeholder", "PORT"),
                 ("x", "f", x), ("y", "f", y)],
            )
        )
    drv_by_macro = {}
    for (m, k), sinks in hard_drv_inputs.items():
        drv_by_macro.setdefault(m, {})[k] = sinks
    for m in range(cfg.n_hard):
        w, h = hard_sizes[m]
        cx, cy = hard_pos[m]
        blocks.append(
            _node(
                f"a{m}", None,
                [("height", "f", h), ("orientation", "placeholder", "N"),
                 ("type", "placeholder", "MACRO"),
                 ("width", "f", w), ("x", "f", cx), ("y", "f", cy)],
            )
        )
        for k, (ox, oy) in enumerate(hard_pins[m]["off"]):
            blocks.append(
                _node(
                    f"a{m}/IP{k}",
                    drv_by_macro.get(m, {}).get(k),
                    [("macro_name", "placeholder", f"a{m}"),
                     ("type", "placeholder", "MACRO_PIN"),
                     ("x", "f", cx + ox), ("x_offset", "f", ox),
                     ("y", "f", cy + oy), ("y_offset", "f", oy)],
                )
            )
    for m in range(cfg.n_soft):
        w, h = soft_sizes[m]
        cx, cy = soft_pos[m]
        blocks.append(
            _node(
                f"Grp_{m}", None,
                [("height", "f", h), ("type", "placeholder", "macro"),
                 ("width", "f", w), ("x", "f", cx), ("y", "f", cy)],
            )
        )
        pin_attrs = lambda: [  # noqa: E731 - soft pins sit at the macro center
            ("macro_name", "placeholder", f"Grp_{m}"),
            ("type", "placeholder", "macro_pin"),
            ("x", "f", cx), ("x_offset", "f", 0.0),
            ("y", "f", cy), ("y_offset", "f", 0.0),
        ]
        blocks.append(_node(f"Grp_{m}/Pinput", None, pin_attrs()))
        for j, sinks in enumerate(soft_outputs.get(m, [])):
            blocks.append(_node(f"Grp_{m}/Poutput_{j}", sinks, pin_attrs()))
    path.write_text("\n".join(blocks) + "\n")


def write_plc(path, cfg, plc, total_area):
    """Write initial.plc: meta header + one line per port/hard/soft module."""
    lines = [
        "# Placement file for Circuit Training",
        f"# Source : synthetic generator ({cfg.name})",
        f"# Columns : {cfg.grid_cols}  Rows : {cfg.grid_rows}",
        f"# Width : {cfg.canvas_w:.3f}  Height : {cfg.canvas_h:.3f}",
        f"# Area : {total_area}",
        "# Project : macro_place_synthetic",
        "# Block : unset_block",
        f"# Routes per micron, hor : {cfg.routes_h:.3f}  ver : {cfg.routes_v:.3f}",
        f"# Routes used by macros, hor : {cfg.macro_routes_h:.3f}  ver : {cfg.macro_routes_v:.3f}",
        "# Smoothing factor : 2",
        "# Overlap threshold : 0.004",
        "#",
    ]
    for idx in plc.port_indices:
        x, y = plc.modules_w_pins[idx].get_pos()
        lines.append(f"{idx} {x:.4f} {y:.4f} - 0")
    for idx in plc.hard_macro_indices + plc.soft_macro_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        fix = 1 if node.get_fix_flag() else 0
        lines.append(f"{idx} {x:.4f} {y:.4f} N {fix}")
    path.write_text("\n".join(lines) + "\n")


# ── Per-benchmark generation ─────────────────────────────────────────────────


def seed_overlap_count(sizes, pos, n):
    """Vectorized count of overlapping hard-macro pairs in the seed."""
    p = np.asarray([pos[i] for i in range(n)])
    s = np.asarray(sizes[:n])
    dx = np.abs(p[:, None, 0] - p[None, :, 0])
    dy = np.abs(p[:, None, 1] - p[None, :, 1])
    mx = (s[:, None, 0] + s[None, :, 0]) / 2
    my = (s[:, None, 1] + s[None, :, 1]) / 2
    ov = (dx < mx) & (dy < my)
    return int((np.triu(ov, 1)).sum())


def generate(cfg):
    rng = np.random.default_rng(cfg.rng_seed)
    case_dir = OUT / "testcases" / cfg.name
    case_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "processed").mkdir(exist_ok=True)
    (OUT / "metadata").mkdir(exist_ok=True)

    hard_sizes = sample_hard_sizes(cfg, rng)
    soft_sizes = sample_soft_sizes(cfg, rng)
    regions = split_regions(0.0, 0.0, cfg.canvas_w, cfg.canvas_h, cfg.n_clusters)

    # cluster assignment proportional to region area
    region_area = np.array([r[2] * r[3] for r in regions])
    p_region = region_area / region_area.sum()
    hard_cluster = rng.choice(cfg.n_clusters, size=cfg.n_hard, p=p_region)
    soft_cluster = rng.choice(cfg.n_clusters, size=cfg.n_soft, p=p_region)

    # fixed macros: pin the largest ones at preset spots
    fixed_mask = np.zeros(cfg.n_hard, dtype=bool)
    hard_pos = {}
    if cfg.n_fixed:
        areas = hard_sizes[:, 0] * hard_sizes[:, 1]
        fixed_ids = list(np.argsort(-areas)[: cfg.n_fixed])
        fixed_mask[fixed_ids] = True
        hard_pos.update(place_fixed(cfg, hard_sizes, fixed_ids))

    # coherent spread placement: used to build the netlist (locality), and as
    # the seed unless seed_style == "random"
    for c, region in enumerate(regions):
        ids = [i for i in np.where(hard_cluster == c)[0] if not fixed_mask[i]]
        hard_pos.update(shelf_pack(rng, ids, hard_sizes, region))
    # nudge movable seeds out of fixed blocks
    for i in range(cfg.n_hard):
        if fixed_mask[i]:
            continue
        for _ in range(20):
            cx, cy = hard_pos[i]
            clash = False
            for j in np.where(fixed_mask)[0]:
                fx, fy = hard_pos[j]
                if (abs(cx - fx) < (hard_sizes[i][0] + hard_sizes[j][0]) / 2
                        and abs(cy - fy) < (hard_sizes[i][1] + hard_sizes[j][1]) / 2):
                    clash = True
                    break
            if not clash:
                break
            hard_pos[i] = random_pos(
                rng, hard_sizes[i], (0.0, 0.0, cfg.canvas_w, cfg.canvas_h)
            )

    soft_pos = {}
    for m in range(cfg.n_soft):
        x0, y0, w_r, h_r = regions[soft_cluster[m]]
        inset = (
            x0 + soft_sizes[m][0] / 2, y0 + soft_sizes[m][1] / 2,
            max(0.01, w_r - soft_sizes[m][0]), max(0.01, h_r - soft_sizes[m][1]),
        )
        soft_pos[m] = (
            rng.uniform(inset[0], inset[0] + inset[2]),
            rng.uniform(inset[1], inset[1] + inset[3]),
        )

    port_pos, port_sides = gen_ports(cfg, rng)
    centers = np.array([(r[0] + r[2] / 2, r[1] + r[3] / 2) for r in regions])
    port_cluster = np.argmin(
        np.linalg.norm(port_pos[:, None, :] - centers[None, :, :], axis=2), axis=1
    )

    hard_pins = make_hard_pins(rng, hard_sizes)
    hard_xy = np.array([hard_pos[i] for i in range(cfg.n_hard)])
    soft_xy = np.array([soft_pos[i] for i in range(cfg.n_soft)])
    port_inputs, hard_drv_inputs, soft_outputs = build_nets(
        cfg, rng, hard_cluster, soft_cluster, port_cluster, hard_pins,
        hard_xy, soft_xy, port_pos,
    )

    # nets were built against the coherent placement; for "random" seeds the
    # written positions are scrambled afterwards so locality structure survives
    if cfg.seed_style == "random":
        full = (0.0, 0.0, cfg.canvas_w, cfg.canvas_h)
        for i in range(cfg.n_hard):
            if not fixed_mask[i]:
                hard_pos[i] = random_pos(rng, hard_sizes[i], full)
        for m in range(cfg.n_soft):
            soft_pos[m] = random_pos(rng, soft_sizes[m], full)

    netlist_path = case_dir / "netlist.pb.txt"
    write_netlist(
        netlist_path, cfg, hard_sizes, hard_pos, fixed_mask, soft_sizes,
        soft_pos, port_pos, port_sides, hard_pins,
        port_inputs, hard_drv_inputs, soft_outputs,
    )

    # parse with the ground-truth evaluator to assign module indices, mark
    # fixed flags, and emit a consistent initial.plc
    from macro_place._plc import PlacementCost

    plc = PlacementCost(str(netlist_path))
    for local_i, idx in enumerate(plc.hard_macro_indices):
        if fixed_mask[local_i]:
            plc.modules_w_pins[idx].set_fix_flag(True)
    total_area = float(
        (hard_sizes[:, 0] * hard_sizes[:, 1]).sum()
        + (soft_sizes[:, 0] * soft_sizes[:, 1]).sum()
    )
    write_plc(case_dir / "initial.plc", cfg, plc, round(total_area, 4))

    # round-trip through the standard loader: validates format + .pt mirror
    from macro_place.loader import load_benchmark

    benchmark, plc2 = load_benchmark(str(netlist_path), str(case_dir / "initial.plc"))
    assert benchmark.num_hard_macros == cfg.n_hard
    assert benchmark.num_soft_macros == cfg.n_soft
    assert int(benchmark.macro_fixed.sum()) == cfg.n_fixed
    benchmark.save(str(OUT / "processed" / f"{cfg.name}.pt"))

    meta = asdict(cfg)
    meta["num_nets"] = int(plc2.net_cnt)
    meta["seed_overlaps"] = seed_overlap_count(hard_sizes, hard_pos, cfg.n_hard)
    (OUT / "metadata" / f"{cfg.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return benchmark, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="generate a single benchmark by name")
    args = parser.parse_args()

    rows = []
    for cfg in SUITE:
        if args.only and cfg.name != args.only:
            continue
        print(f"generating {cfg.name}...", flush=True)
        benchmark, meta = generate(cfg)
        rows.append((cfg.name, benchmark, meta))
        print(
            f"  {benchmark}  nets={meta['num_nets']} "
            f"grid={cfg.grid_rows}x{cfg.grid_cols} fixed={cfg.n_fixed} "
            f"ports={cfg.n_ports} seed_overlaps={meta['seed_overlaps']}"
        )
    print(f"\ndone: {len(rows)} benchmarks under {OUT / 'testcases'}")


if __name__ == "__main__":
    main()
