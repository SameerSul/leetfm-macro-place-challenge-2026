# Why The Deleted Ranker Could Improve Proxy - Archived

This document is historical. The proxy-path ML ranker and all `src/placer/ml/`
code have been deleted from active code.

The original idea was sound for the old proxy optimizer: exact scoring was the
oracle, and the model only chose which candidates to score first under a fixed
deadline. It could improve output by reaching useful exact-gated moves earlier,
without ever replacing the proxy accept gate.

That reasoning no longer applies to the current production flow because the R2
candidate search it accelerated has been removed. The hierarchy path uses
grouped DREAMPlace, cluster-consecutive legalization, region-locked relief, and
bounded coldspot tightening instead.
