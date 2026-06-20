#!/usr/bin/env python3
"""Compare heuristic and GNN-ranked hierarchy trace files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _sum_trace(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    events = Counter(str(row.get("event", "")) for row in rows)
    passes: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "accepts": 0.0,
            "proxy_delta": 0.0,
            "proxy_before": 0.0,
            "proxy_after": 0.0,
        }
    )
    relocation: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "accepts": 0.0,
            "verify_scores": 0.0,
            "proxy_delta": 0.0,
            "proposal_count": 0.0,
            "legal_count": 0.0,
            "candidate_samples": 0.0,
            "gnn_scored_samples": 0.0,
            "gnn_error_samples": 0.0,
        }
    )
    final = None
    for row in rows:
        event = row.get("event")
        if event == "hier_pass_result":
            name = str(row.get("hierarchy_pass", "unknown"))
            item = passes[name]
            item["count"] += 1.0
            item["accepts"] += float(row.get("accepts") or 0.0)
            item["proxy_delta"] += float(row.get("proxy_delta") or 0.0)
            item["proxy_before"] = float(row.get("proxy_before") or 0.0)
            item["proxy_after"] = float(row.get("proxy_after") or 0.0)
        elif event == "hier_relocation_result":
            name = f"{row.get('kind', 'unknown')}:{row.get('field', 'unknown')}"
            item = relocation[name]
            item["count"] += 1.0
            item["accepts"] += float(row.get("accepts") or 0.0)
            item["verify_scores"] += float(row.get("verify_scores") or 0.0)
            item["proxy_delta"] += float(row.get("final_proxy") or 0.0) - float(
                row.get("initial_proxy") or 0.0
            )
        elif event == "hier_relocation_candidates":
            name = f"{row.get('kind', 'unknown')}:{row.get('field', 'unknown')}"
            item = relocation[name]
            item["proposal_count"] += float(row.get("proposal_count") or 0.0)
            item["legal_count"] += float(row.get("legal_count") or 0.0)
            samples = row.get("candidates") or []
            item["candidate_samples"] += float(len(samples))
            item["gnn_scored_samples"] += float(
                sum(1 for c in samples if c.get("gnn_score") is not None)
            )
            item["gnn_error_samples"] += float(
                sum(1 for c in samples if c.get("gnn_rank_error") is not None)
            )
        elif event == "hier_final":
            final = row
    return {
        "path": str(path),
        "events": dict(sorted(events.items())),
        "final": final or {},
        "passes": dict(sorted(passes.items())),
        "relocation": dict(sorted(relocation.items())),
    }


def _diff_maps(base: dict[str, Any], ranked: dict[str, Any]) -> dict[str, Any]:
    out = {}
    keys = sorted(set(base) | set(ranked))
    for key in keys:
        b = base.get(key, {})
        r = ranked.get(key, {})
        vals = {}
        for metric in sorted(set(b) | set(r)):
            bv = float(b.get(metric, 0.0) or 0.0)
            rv = float(r.get(metric, 0.0) or 0.0)
            vals[metric] = {"heuristic": bv, "gnn": rv, "delta": rv - bv}
        out[key] = vals
    return out


def compare(heuristic: Path, gnn: Path) -> dict[str, Any]:
    h = _sum_trace(heuristic)
    g = _sum_trace(gnn)
    h_proxy = float(h["final"].get("proxy") or 0.0)
    g_proxy = float(g["final"].get("proxy") or 0.0)
    return {
        "heuristic": h,
        "gnn": g,
        "final_proxy": {
            "heuristic": h_proxy,
            "gnn": g_proxy,
            "delta": g_proxy - h_proxy,
        },
        "pass_deltas": _diff_maps(h["passes"], g["passes"]),
        "relocation_deltas": _diff_maps(h["relocation"], g["relocation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heuristic", type=Path, required=True)
    parser.add_argument("--gnn", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = compare(args.heuristic, args.gnn)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    proxy = report["final_proxy"]
    print(
        "Trace pair: "
        f"heuristic={proxy['heuristic']:.4f} "
        f"gnn={proxy['gnn']:.4f} "
        f"delta={proxy['delta']:+.4f}"
    )
    for name, vals in report["relocation_deltas"].items():
        accepts = vals.get("accepts", {})
        verify = vals.get("verify_scores", {})
        samples = vals.get("gnn_scored_samples", {})
        print(
            f"  {name}: accepts {accepts.get('heuristic', 0):.0f}->"
            f"{accepts.get('gnn', 0):.0f}, exact "
            f"{verify.get('heuristic', 0):.0f}->{verify.get('gnn', 0):.0f}, "
            f"gnn_samples={samples.get('gnn', 0):.0f}"
        )


if __name__ == "__main__":
    main()
