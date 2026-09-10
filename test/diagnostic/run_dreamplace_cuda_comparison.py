"""Compare DREAMPlace backends through the unchanged placement pipeline.

Each output directory starts with a fresh seed cache. Search deadlines are
disabled by default, retaining deterministic quotas; --normal keeps deadlines.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import run_speed_comparison as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["DREAMPLACE_GPU"] = str(args.gpu)
    from dreamplace_bridge import run_bridge as bridge

    original = bridge.run_dreamplace
    calls = []

    def measured(*positional, **kwargs):
        seed = kwargs.get("seed_name") or "dreamplace"
        kwargs["scratch_root"] = str(args.out / "dreamplace" / seed)
        kwargs["keep_log"] = True
        start = time.perf_counter()
        result = original(*positional, **kwargs)
        calls.append(
            dict(
                benchmark=Path(positional[0]).name,
                seed=seed,
                gpu=args.gpu,
                elapsed_s=time.perf_counter() - start,
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
