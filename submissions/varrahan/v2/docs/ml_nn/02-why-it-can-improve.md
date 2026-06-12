# Why a model trained on our own placer can still beat it

## The apparent paradox

The training data comes from our own placer's runs. So how can a model trained on
it produce placements *better* than our placer? At best it should match us — and
it can't out-predict the scorer it learned from. This worry is half right, and
the half that's wrong is the important one.

## Resolution: the model competes with our *ordering*, not our *scorer*

The model never decides whether a move is accepted — the exact proxy gate does.
The model only decides **which candidates are worth exact-scoring**. Its labels
are the true `score_gain` from the exact proxy, so:

> The exact scorer is the **oracle**. The model is a cheap approximation of the
> oracle used for triage. It can't be "better than" the scorer — that's not its
> job. Its job is to beat the **hand-coded candidate ordering** under a deadline.

"Trained on our own runs" is fine because the *label* (did this move actually
lower the proxy?) is ground truth from the objective, not from our placer's
opinion.

## The ceiling is convergence — and we don't reach it

The relevant ceiling is the **exhaustive search run to full convergence**: score
every candidate, every round, until no improving move exists. The model's best
case is that ceiling.

**Our placer does not reach that ceiling on the hard benchmarks.** It's
deadline-bound — on ibm12/14/15/17/18 the exact score is so slow that R2 gets a
handful of rounds and evaluates a fraction of the candidates before time runs
out. It stops *short* of its own ceiling. That gap — between our truncated output
and the converged result — is what the ranker closes, two ways, both inside the
same wall-clock budget:

1. **Front-loading.** Today candidates are tried in a hand-coded order
   (hottest-first, nearest-target-first). That order is a guess, and it's wrong
   often enough that productive moves sit late in the queue and never get scored
   before the deadline. The ranker scores the productive ones first → more
   accepts realized before time runs out.
2. **Freed budget → more rounds.** Skipping losers frees scoring time, and on
   these benchmarks freed time converts to additional R2 rounds — reaching
   deeper than our placer currently has time to.

Every individual accept is still the exact gate's call. The model just gets us
closer to convergence inside the deadline. It does **not** expand the reachable
placement set — only the *budget-reachable* set. The placements it reaches are
ones the exhaustive search could reach given infinite time.

## Where the intuition is exactly right

On benchmarks that already **converge with budget to spare** (ibm01, ibm09), the
ranker offers nothing and can only *hurt* via **under-improvement**: if it
mis-ranks and prunes the true-best candidate, the exact gate never sees it.
Ceiling = current output, downside only. So:

- Gains concentrate on the **slow, budget-bound** benchmarks.
- We validate **per-benchmark**, not on the average (the history is full of
  "win ibm04 / lose ibm09" reverts).
- **recall@K** matters — it bounds how often we drop the true-best move.

## The legitimate version of the worry: distribution shift

The labels are "given a state our *current* placer visits, did this candidate
improve?" Once the ranker changes the order of acceptance, the search visits
*different* states, where the model is extrapolating. Two things mitigate it:

- Much of what it learns is a property of the **proxy itself** — "high-degree
  macro, hot source cell, cold low-density target → tends to improve" holds
  regardless of trajectory — so it transfers.
- Where it doesn't transfer, that's the real risk, fixed by a **DAgger** cycle:
  run the v0 ranker, collect traces from the states *it* induces, retrain on the
  union. A single offline model usually underperforms its own validation until
  this loop is closed.

## One-line summary

The model can beat our placer's **output** without ever beating the exact scorer
it learned from, because our placer doesn't reach its own ceiling under the
deadline — the model just spends the fixed scoring budget closer to optimally.
The "free energy" comes from the search currently being left unfinished, not from
out-predicting the scorer. It's not a perpetual-motion machine.
