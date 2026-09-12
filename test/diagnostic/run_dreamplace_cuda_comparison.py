"""Compare DREAMPlace backends through the unchanged placement pipeline.

Each output directory starts with a fresh seed cache. Search deadlines are
disabled by default, retaining deterministic quotas; --normal keeps deadlines.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import run_speed_comparison as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normal", action="store_true")
    parser.add_argument(
        "--capture-final", action="store_true", help="save hierarchy inputs for replay"
    )
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    from utils import constants as const

    const.DREAMPLACE_GPU = bool(args.gpu)
    from dreamplace_bridge import run_bridge as bridge

    files = sorted((run.ROOT / "src").rglob("*.py")) + [Path(__file__)]
    (args.out / "source.json").write_text(
        json.dumps(
            dict(
                gpu=args.gpu,
                normal=args.normal,
                capture_final=args.capture_final,
                files={
                    str(p.relative_to(run.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            ),
            indent=2,
        )
        + "\n"
    )
    if args.capture_final:
        from run_algorithm_comparison import capture_final_inputs

        capture_final_inputs(args.out)

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
