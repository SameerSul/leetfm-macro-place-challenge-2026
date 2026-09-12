"""Paired GPU placement experiments with fresh caches and captured final contracts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import run_speed_comparison as run
from run_algorithm_comparison import capture_final_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "rudy"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["DREAMPLACE_GPU"] = str(args.gpu)
    from dreamplace_bridge import run_bridge as bridge
    from dreamplace_bridge.pb_to_bookshelf import _sanitize

    files = sorted((run.ROOT / "src").rglob("*.py")) + [
        Path(__file__),
        Path(__file__).with_name("dreamplace_rudy.py"),
    ]
    hashes = {
        str(p.relative_to(run.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (args.out / "source.json").write_text(
        json.dumps(dict(mode=args.mode, gpu=args.gpu, normal=args.normal, files=hashes), indent=2)
        + "\n"
    )
    capture_final_inputs(args.out)
    original = bridge.run_dreamplace
    calls = []
    if args.mode == "rudy":
        bridge.DREAMPLACE_PLACER = Path(__file__).with_name("dreamplace_rudy.py")
    original_convert = bridge.convert

    def convert(*positional, **kwargs):
        result = original_convert(*positional, **kwargs)
        plc = kwargs["plc"]
        out, design = Path(positional[1]), kwargs["design"]
        metadata = dict(
            grid=[plc.grid_col, plc.grid_row],
            canvas=[plc.width, plc.height],
            scale=float((out / f"{design}.scale").read_text()),
            horizontal_capacity=plc.hroutes_per_micron,
            vertical_capacity=plc.vroutes_per_micron,
            horizontal_blockage=plc.hrouting_alloc,
            vertical_blockage=plc.vrouting_alloc,
            hard_names=[
                _sanitize(plc.modules_w_pins[i].get_name()) for i in plc.hard_macro_indices
            ],
        )
        (out / f"{design}.routing.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return result

    bridge.convert = convert

    def measured(*positional, **kwargs):
        seed = kwargs.get("seed_name") or "dreamplace"
        kwargs["scratch_root"] = str(args.out / "dreamplace" / seed)
        kwargs["keep_log"] = True
        start = time.perf_counter()
        result = original(*positional, **kwargs)
        calls.append(
            dict(
                benchmark=Path(positional[0]).name, seed=seed, elapsed_s=time.perf_counter() - start
            )
        )
        (args.out / "dreamplace_calls.json").write_text(json.dumps(calls, indent=2) + "\n")
        return result

    bridge.run_dreamplace = measured
    sys.argv = [__file__, "--out", str(args.out)]
    if not args.normal:
        sys.argv.append("--no-deadlines")
    sys.argv.extend(args.names)
    run.main()


if __name__ == "__main__":
    main()
