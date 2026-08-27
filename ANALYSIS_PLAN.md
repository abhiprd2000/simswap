# Analysis plan

Pre-registered 2026-08-14, before any real tau2-bench data exists. Not
edited after real data lands. 

## Grid

4 agents x 4 user models x 20 tasks x 2 seeds = 640 runs.

## Minimum detectable effect

Minimum detectable interaction magnitude at this grid: **delta = 0.12**
on the action-accuracy scale. Established on synthetic data with
`detectability.py` before any real run (see `tables/detectability.csv`
and the paper appendix). Below delta = 0.12 we should not expect the
tests in this plan to reliably separate a real inversion from noise at
this grid size.

Seeds fixed at 2, not 3: the sweep across 20x2 and 20x3 tasks found
identical minimum detectable delta (0.12) for both. The remaining
variance is between tasks, not within a task across seeds, so a third
seed buys nothing at this grid size. GPU-hours go to more tasks before
more seeds if the real data turns out underpowered.

## Primary claim: rank inversion

A rank inversion between two agents under two user models is reported as
a supported finding **only if all three hold**:

1. The omnibus interaction LRT (`outcome ~ agent * user_model + (1|task)`
   vs `outcome ~ agent + user_model + (1|task)`) is significant.
2. At least one targeted contrast (paired sign-flip test on the
   double-difference, Holm-corrected across all 36 agent-pair x
   simulator-pair contrasts) survives correction.
3. The LRT and the targeted contrast agree on which agent pair and which
   simulator pair moved.

Corroboration between the omnibus test and a targeted contrast is the
bar, not any single p-value from either test alone.

## Lone Holm-significant contrasts

A targeted contrast that survives Holm correction while the omnibus LRT
is not significant is reported as a **null**, not a finding. Across 36
simultaneous contrasts at nominal 5% Holm-corrected error, we expect
roughly one such contrast by chance even when nothing is real. The
detectability sweep produced exactly this pattern at several synthetic
grids (one recurring Holm-significant contrast unrelated to the planted
effect, present at every tested delta, absent at LRT significance) - the
paper says so explicitly rather than treating an unreplicated single
contrast as evidence.

## Family-match term

The `family_match` term in the mixed-effects model
(`outcome ~ agent + user_model + family_match + (1|task)`) is secondary
and underpowered by design: family-match cells are a small slice of the
16-cell grid. It is reported alongside the main effects for completeness.
It cannot be promoted to a headline claim after the fact, regardless of
what it shows in the real data.

## Rank permutation test

`rank_permutation_test_conservative.csv` is kept as a conservative,
secondary check only. It is a discrete rank statistic (minimum nonzero
value = 1/C(4,2) = 1/6 with 4 agents) and is underpowered by construction
for a panel this small - it discards effect magnitude. It does not factor
into the primary claim above in either direction.

## tau2-bench version

tau2-bench version and commit are pinned and recorded per run
(`tau2_version`, `tau2_commit` in schema.py). Never upgraded mid-sweep -
v1.0.1 changed grading, so runs across versions are not comparable.
