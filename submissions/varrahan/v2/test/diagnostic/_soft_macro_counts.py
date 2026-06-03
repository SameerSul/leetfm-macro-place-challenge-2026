#!/usr/bin/env python3
"""Print soft macro counts for all 17 IBM benchmarks."""
import sys, torch
sys.path.insert(0, '.')
from macro_place.benchmark import Benchmark

benchmarks = ['ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08','ibm09',
              'ibm10','ibm11','ibm12','ibm13','ibm14','ibm15','ibm16','ibm17','ibm18']
for b in benchmarks:
    bm = Benchmark.load(f'benchmarks/processed/public/{b}.pt')
    movable = int(bm.get_movable_mask().numpy()[bm.num_hard_macros:].sum())
    print(f"{b}: hard={bm.num_hard_macros:4d}  soft={bm.num_soft_macros:4d}  soft_movable={movable:4d}")
