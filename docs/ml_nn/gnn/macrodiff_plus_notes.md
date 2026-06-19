# MacroDiff+ Design Notes

Reference paper:

- [Physics-Guided Geometric Diffusion for Macro Placement Generation](https://arxiv.org/pdf/2605.16451)

This paper is relevant because it combines topology-aware graph learning with
physical placement constraints. It should inform our G4 graph-model design, but
it should not redefine our subsystem as a full generative placer. The expanded
target is a hierarchy-flow assistant, documented in
[expansion-plan.md](expansion-plan.md), not diffusion-based coordinate
generation.

## What Applies To Our GNN

Borrow these ideas:

- **Heterogeneous macro-net graph.** Represent macros and nets as distinct node
  types instead of only projecting nets into macro-macro clique edges.
- **Pin-offset edge features.** Preserve per-pin offsets on macro-net edges so
  the model can learn wirelength effects for large macros more accurately than
  center-to-center connectivity.
- **Static and dynamic feature split.** Keep invariant design features separate
  from placement-state features:
  - static: macro size, chip dimensions, net degree, pin offsets;
  - dynamic: current position, candidate position, net HPWL / wirelength
    pressure, congestion or density fields when available.
- **Topology plus geometry.** Combine graph embeddings with scalar candidate
  geometry features. For us this means source/target/cluster embeddings plus the
  candidate feature table from Stage G2.
- **Physical constraints stay explicit.** Learned scores do not replace overlap,
  bounds, fixed-macro, hierarchy-region, hierarchy-quality, or exact-proxy
  gates.
- **Auxiliary physical heads.** Use the same physical-awareness principle to
  predict proxy delta, hierarchy-quality delta, rejection reason, and scoring
  risk for existing hierarchy candidates.

## What Does Not Apply Directly

Do not implement these as part of the current GNN ranker:

- diffusion sampling;
- full-placement coordinate generation;
- replacing hierarchy operators with learned placement;
- training the model to denoise complete placements;
- using learned gradients as an accept gate.

Those are useful research directions, but they conflict with the current
production contract: the GNN is a candidate ranker inside the hierarchy flow.
Candidate proposal, operator selection, region guidance, soft-role guidance, and
budget allocation are allowed follow-ons only when they remain gated inside the
existing implementation.

## Required Design Changes For G4

The first G4 graph model should extend the Stage-G2 graph format with:

- net nodes;
- macro-net edges;
- edge features for pin offset, net weight, and pin role if available;
- net-node features for degree, normalized HPWL, and optional congestion/density
  pressure;
- candidate-move context edges or candidate side features linking source and
  target.

The model should remain small:

- 2-3 message-passing layers;
- hidden size 32 or 64;
- CPU inference as the default target.

## Acceptance Boundary

MacroDiff+ reports benefits from combining topology, geometry, and explicit
physics guidance. Our equivalent production boundary is:

- topology: hetero macro-net GNN;
- geometry: candidate scalar features and optional spatial-neighbor edges;
- physics: existing deterministic legality, hierarchy, and exact-proxy gates.

The paper can justify richer graph features, but it does not justify bypassing
the placement gates.
