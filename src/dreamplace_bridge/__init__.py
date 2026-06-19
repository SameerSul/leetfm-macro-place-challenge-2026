"""DREAMPlace bridge for the v2 placer.

Three-step pipeline:
    pb_to_bookshelf.convert(...)       - TILOS pb.txt → 5-file Bookshelf bundle
    run_bridge.run_dreamplace(...)     - Bookshelf → DREAMPlace global → positions
    bookshelf_to_pb.read_dreamplace_positions(...) - back-convert to TILOS coords

Used by `placer.pipeline.macro_placer` for the required grouped DREAMPlace
stage in the hierarchy-only production path.
"""
