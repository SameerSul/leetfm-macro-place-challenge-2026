"""Diagnostic DREAMPlace entrypoint: physical-net RUDY plus moving hard blockages.

Runs in DREAMPlace's build Python. No installed or production files are patched.
The additive blockage field is an independent surrogate, not the exact evaluator.
"""

import json
import logging
from pathlib import Path
import sys


def blockage_map(lower, upper, x_edges, y_edges):
    """Fraction of each rectangular bin covered by the supplied macro footprints."""
    import torch

    dx = (
        torch.minimum(upper[:, 0, None], x_edges[None, 1:])
        - torch.maximum(lower[:, 0, None], x_edges[None, :-1])
    ).clamp_min(0)
    dy = (
        torch.minimum(upper[:, 1, None], y_edges[None, 1:])
        - torch.maximum(lower[:, 1, None], y_edges[None, :-1])
    ).clamp_min(0)
    return (dx.T @ dy) / ((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))


def physical_net_mask(names):
    return [not (n.decode() if isinstance(n, bytes) else str(n)).startswith("grp") for n in names]


def main():
    import torch
    import Params
    import Placer
    import PlaceObj
    from dreamplace.ops.rudy.rudy import Rudy

    config = Path(sys.argv[1])
    metadata = json.loads(config.with_suffix(".routing.json").read_text())
    params = Params.Params()
    params.load(str(config))
    params.routability_opt_flag = 1
    params.adjust_pin_area_flag = 0
    params.adjust_nctugr_area_flag = 0
    params.adjust_rudy_area_flag = 1
    params.max_num_area_adjust = 1
    # The ordinary 0.15 threshold is never reached on the captured IBM10 solve
    # (minimum 0.214). Test one earlier adjustment, below 0.30, per native stage.
    params.node_area_adjust_overflow = 0.30
    params.route_num_bins_x, params.route_num_bins_y = metadata["grid"]
    # Bookshelf coordinates are scaled by 1000, so tracks per unit scale inversely.
    params.unit_horizontal_capacity = metadata["horizontal_capacity"] / metadata["scale"]
    params.unit_vertical_capacity = metadata["vertical_capacity"] / metadata["scale"]
    calls = []
    adjustments = []
    original_adjust = PlaceObj.PlaceObj.build_adjust_node_area

    def build_adjust(self, *args):
        op = original_adjust(self, *args)

        def adjust(*values):
            result = op(*values)
            adjustments.append([bool(value) for value in result])
            return result

        return adjust

    PlaceObj.PlaceObj.build_adjust_node_area = build_adjust

    def build(self, params, placedb, data):
        device, dtype = data.net_weights.device, data.net_weights.dtype
        physical = torch.tensor(physical_net_mask(placedb.net_names), device=device)
        weights = data.net_weights.clone() * physical
        hard_names = set(metadata["hard_names"])
        indices = torch.tensor(
            [i for i, name in enumerate(placedb.node_names) if name.decode() in hard_names],
            device=device,
            dtype=torch.long,
        )
        width, height = metadata["canvas"]
        width, height = width * metadata["scale"], height * metadata["scale"]
        nx, ny = metadata["grid"]
        x_edges = torch.linspace(0, width, nx + 1, device=device, dtype=dtype)
        y_edges = torch.linspace(0, height, ny + 1, device=device, dtype=dtype)
        hblock = torch.zeros((nx, ny), device=device, dtype=dtype)
        vblock = torch.zeros_like(hblock)
        op = Rudy(
            netpin_start=data.flat_net2pin_start_map,
            flat_netpin=data.flat_net2pin_map,
            net_weights=weights,
            xl=0,
            yl=0,
            xh=width,
            yh=height,
            num_bins_x=nx,
            num_bins_y=ny,
            unit_horizontal_capacity=params.unit_horizontal_capacity,
            unit_vertical_capacity=params.unit_vertical_capacity,
            deterministic_flag=1,
            initial_horizontal_utilization_map=hblock,
            initial_vertical_utilization_map=vblock,
        )
        record = dict(
            physical_nets=int(physical.sum()),
            excluded_nets=int((~physical).sum()),
            hard_macros=len(indices),
            device=str(device),
            calls=0,
        )
        calls.append(record)

        def route(pos):
            with torch.no_grad():
                count = pos.numel() // 2
                current_size = torch.stack(
                    (data.node_size_x[indices], data.node_size_y[indices]), 1
                )
                original_size = torch.stack(
                    (data.original_node_size_x[indices], data.original_node_size_y[indices]), 1
                )
                center = torch.stack((pos[indices], pos[count + indices]), 1) + current_size / 2
                occupancy = blockage_map(
                    center - original_size / 2, center + original_size / 2, x_edges, y_edges
                )
                hblock.copy_(
                    occupancy * metadata["horizontal_blockage"] / metadata["horizontal_capacity"]
                )
                vblock.copy_(
                    occupancy * metadata["vertical_blockage"] / metadata["vertical_capacity"]
                )
                record["calls"] += 1
                return op(self.op_collections.pin_pos_op(pos))

        return route

    PlaceObj.PlaceObj.build_route_utilization_map = build
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="[%(levelname)s] %(message)s", force=True
    )
    metrics = Placer.place(params, params.global_place_stages[0]["learning_rate"])

    def overflows(value):
        if isinstance(value, list):
            return [x for item in value for x in overflows(item)]
        if getattr(value, "overflow", None) is not None:
            return [float(value.overflow.min())]
        return []

    minimum_overflow = min(overflows(metrics), default=None)
    logging.info("minimum observed overflow: %s", minimum_overflow)
    config.with_suffix(".routing_stats.json").write_text(
        json.dumps(
            dict(
                routes=calls,
                adjustments=adjustments,
                minimum_overflow=minimum_overflow,
                adjustment_threshold=params.node_area_adjust_overflow,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
