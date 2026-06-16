# Ranker Selection Mechanism - Archived

This document is historical. The proxy-path ML ranker and all `src/placer/ml/`
code have been deleted from active code.

The deleted selector predicted which source macros to attempt, which target
cells or swap partners to exact-score, and which candidates were likely to
survive the exact proxy gate.

It was a triage layer for the old R2 loop, not a placement generator. Since the
R2 loop has been removed from production, this mechanism is no longer runnable.
