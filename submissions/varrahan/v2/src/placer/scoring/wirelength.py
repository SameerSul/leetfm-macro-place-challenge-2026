"""Vectorized wirelength helpers."""

import numpy as np

from placer.plc.placement import _ensure_pos_cache

def _build_macro_pin_map(plc):
    """Cache MACRO_NAME -> [pin_indices] on plc (mirrors objective._set_placement's
    one-time build, but built eagerly here so the fast path doesn't fork on the
    hasattr check every call)."""
    if hasattr(plc, "_macro_pin_map"):
        return plc._macro_pin_map
    pin_map: "dict[str, list[int]]" = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
            name = mod.get_macro_name()
            pin_map.setdefault(name, []).append(idx)
    plc._macro_pin_map = pin_map
    return pin_map


def _build_wl_cache(plc):
    """Precompute per-pin arrays used by the vectorized wirelength.

    For each net (in plc.nets.keys() insertion order), record:
      - per-pin ref_node_idx (index into plc.modules_w_pins)
      - per-pin x_offset, y_offset
    Plus per-net weight (from the driver pin) and reduceat boundaries.

    The unified representation: PORT pins use ref_node_idx = port_idx + offset 0.
    Macro pins use ref_node_idx = parent_macro_idx + pin's stored offset.
    Either way, pin_pos = node_pos[ref_idx] + offset; this matches plc's
    private __get_pin_position semantics exactly.
    """
    if hasattr(plc, "_wl_vec_cache"):
        return plc._wl_vec_cache

    ref_idx_list: "list[int]" = []
    x_off_list: "list[float]" = []
    y_off_list: "list[float]" = []
    net_starts: "list[int]" = []
    net_weights: "list[float]" = []
    cursor = 0
    skipped_nets = 0

    name_to_idx = plc.mod_name_to_indices
    mods = plc.modules_w_pins

    def _pin_info(pin_idx: int):
        pin = mods[pin_idx]
        ptype = pin.get_type()
        if ptype == "PORT":
            return pin_idx, 0.0, 0.0
        if ptype == "MACRO_PIN":
            parent_name = pin.get_macro_name()
            ref_idx = name_to_idx.get(parent_name, -1)
            if ref_idx == -1:
                return None
            return ref_idx, float(getattr(pin, "x_offset", 0.0)), float(getattr(pin, "y_offset", 0.0))
        return None

    for driver_pin_name, sink_pin_names in plc.nets.items():
        driver_idx = name_to_idx.get(driver_pin_name)
        if driver_idx is None:
            skipped_nets += 1
            continue
        driver_info = _pin_info(driver_idx)
        if driver_info is None:
            skipped_nets += 1
            continue
        driver_pin = mods[driver_idx]
        try:
            weight = float(driver_pin.get_weight())
        except Exception:
            weight = 1.0

        local_pins = [driver_info]
        for sink_name in sink_pin_names:
            sink_idx = name_to_idx.get(sink_name)
            if sink_idx is None:
                continue
            info = _pin_info(sink_idx)
            if info is None:
                continue
            local_pins.append(info)
        if len(local_pins) < 1:
            skipped_nets += 1
            continue

        net_starts.append(cursor)
        net_weights.append(weight)
        for r, xo, yo in local_pins:
            ref_idx_list.append(r)
            x_off_list.append(xo)
            y_off_list.append(yo)
        cursor += len(local_pins)

    ref_idx_arr = np.asarray(ref_idx_list, dtype=np.int64)
    x_off_arr = np.asarray(x_off_list, dtype=np.float64)
    y_off_arr = np.asarray(y_off_list, dtype=np.float64)
    net_starts_arr = np.asarray(net_starts, dtype=np.int64)
    net_weights_arr = np.asarray(net_weights, dtype=np.float64)

    # Unique ref_node indices (gather destinations) + inverse mapping into
    # pin-flat order. Lets us pull current node positions in one pass and
    # then scatter via numpy indexing.
    unique_ref, inv = np.unique(ref_idx_arr, return_inverse=True)

    # Per-pin → net-index mapping for incremental scoring (touched-net
    # selection given a moved macro).
    pin_to_net = (
        np.searchsorted(net_starts_arr, np.arange(cursor), side="right") - 1
    ).astype(np.int64)
    # Per-net pin lengths (cursor as sentinel for the last net's end).
    net_ends = np.empty_like(net_starts_arr)
    net_ends[:-1] = net_starts_arr[1:]
    net_ends[-1] = cursor
    net_lengths = (net_ends - net_starts_arr).astype(np.int64)


    cache = {
        "ref_idx": ref_idx_arr,
        "ref_inv": inv.astype(np.int64),
        "unique_ref": unique_ref,
        "x_off": x_off_arr,
        "y_off": y_off_arr,
        "net_starts": net_starts_arr,
        "net_ends": net_ends,
        "net_lengths": net_lengths,
        "net_weights": net_weights_arr,
        "pin_to_net": pin_to_net,
        "n_pins": cursor,
        "n_nets": len(net_starts),
    }
    plc._wl_vec_cache = cache
    return cache


def _vectorized_wirelength(plc) -> float:
    """Drop-in numpy replacement for plc.get_wirelength().

    Iterates plc.nets in insertion order (matching scalar semantics), computes
    per-net (max-min) HPWL in vector form via np.minimum/maximum.reduceat, and
    sums in float64. Tiny FP differences vs the scalar loop are possible but
    irrelevant at proxy-cost granularity.
    """
    cache = _build_wl_cache(plc)
    if cache["n_nets"] == 0:
        return 0.0
    unique_ref = cache["unique_ref"]
    pos_cache = _ensure_pos_cache(plc)
    node_x = pos_cache[unique_ref, 0]
    node_y = pos_cache[unique_ref, 1]
    inv = cache["ref_inv"]
    pin_x = node_x[inv] + cache["x_off"]
    pin_y = node_y[inv] + cache["y_off"]
    starts = cache["net_starts"]
    max_x = np.maximum.reduceat(pin_x, starts)
    min_x = np.minimum.reduceat(pin_x, starts)
    max_y = np.maximum.reduceat(pin_y, starts)
    min_y = np.minimum.reduceat(pin_y, starts)
    per_net = cache["net_weights"] * ((max_x - min_x) + (max_y - min_y))
    return float(per_net.sum())


def _patch_plc_wirelength(plc) -> None:
    """Install the vectorized wirelength on this plc instance (idempotent)."""
    if getattr(plc, "_wl_vec_installed", False):
        return
    # Bind as a bound method via lambda to keep the plc API: plc.get_wirelength()
    plc.get_wirelength = lambda _plc=plc: _vectorized_wirelength(_plc)
    plc._wl_vec_installed = True
