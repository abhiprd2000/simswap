# Camera-ready final number pass

Written 2026-10-18. Part B dropped (sandbox has no outbound network -
see `PROGRESS.md`). This is the final numbers document for the
camera-ready: **component-level classifier (`component_level_grounding.py`)
unless stated otherwise** - the as-submitted (default `failure_analysis.py`)
value is always given too, for the disclosure note. Source of truth:
`src/paper_numbers.py`, writes `paper_numbers_raw.json`. No existing
analysis script's default behavior was changed; `paper/PAPER.md` and
`ANALYSIS_PLAN.md` were not edited.

Numbering: T1-T5 as defined in `docs/PAPER_NUMBERS.md`'s A10 section
(T1 conditional rate, T2 per-task, T3 provenance, T4 exposure, T5 schema
conformance; §3.5 post-error has no table number). This is a relabeling
for these documents only, not yet reflected in `paper/PAPER.md` or
`README.md`.

## T1 - conditional ungrounded rate, both agents

| | n/d | % | tasks contributing | as-submitted |
|---|---|---|---|---|
| Qwen | 14/15 | 93.3% | 2 of 3 (task0 contributes 0/0 - pool never fills) | identical (14/15, 93.3%) |
| Llama | 43/95 | 45.3% | 3 of 3 | 45/77 (58.4%) |
| **pooled gap** | - | **48.1pp** | - | 34.9pp |

Source: `paper_numbers.py` (reuses `failure_analysis.analyze_conversation` +
`component_level_grounding.analyze_conversation_component_parsed`, pooled
via `matched_comparison.pool()`).

## T2 - per task 0/1/3

| task | Qwen n/d (%) | Llama n/d (%) | gap | as-sub. Qwen | as-sub. Llama | as-sub. gap |
|---|---|---|---|---|---|---|
| task0 | n/d (pool never fills) | 22/50 (44.0%) | n/d | n/d | 22/38 (57.9%) | n/d |
| task1 | 6/6 (100.0%) | 15/35 (42.9%) | **57.1pp** | 6/6 (100.0%) | 15/29 (51.7%) | 48.3pp |
| task3 | 8/9 (88.9%) | 6/10 (60.0%) | **28.9pp** | 8/9 (88.9%) | 8/10 (80.0%) | 8.9pp |

**Permutation test** (per-task FLAT diffs, Qwen-Llama), exact enumeration
of all 2^3=8 sign assignments:

| | component-level | as-submitted |
|---|---|---|
| per-task diffs | +0.549, +0.556, +0.338 | +0.410, +0.467, +0.137 |
| all 3 same sign | yes (Qwen higher) | yes (Qwen higher) |
| **one-sided p** | **0.125** | 0.125 (unchanged) |
| **two-sided p** | **0.25** | 0.25 (unchanged) |

**Corrected 2026-10-24**: this was previously reported as 0.111/0.222,
using p=(b+1)/(n+1). That correction is for an APPROXIMATE permutation
test built from a finite random sample of sign assignments (it
guarantees a nonzero p-value when the sample might miss the true
tail). This test enumerates all 2^3=8 sign assignments EXHAUSTIVELY,
not a sample - the observed assignment is already one of the 8
outcomes, so it belongs in both the numerator and denominator like any
other: p = b/n with b including the observed assignment. One-sided:
b=1 (only the all-positive assignment reaches the observed sum, since
all three diffs are positive), p=1/8=0.125. Two-sided: b=2 (the
all-positive and all-negative assignments both reach the observed
magnitude), p=2/8=0.25. Floor of this test at n=3 pairs is **1/8**,
not 1/9. Fixed in `src/permutation_test.py` (now exposes a shared
`exact_perm_test()` function) and `src/paper_numbers.py` (imports it,
no longer a separate reimplementation - the two call sites can't drift
apart again). Locked down by
`test_exact_permutation_p_value_uses_full_enumeration_not_sample_correction`.

## T3 - provenance, flat AND conditional, both agents

| | user | policy | invented | as-submitted (same slot) |
|---|---|---|---|---|
| Qwen FLAT (54 ungrounded) | 43 (79.6%) | 0 (0.0%) | 11 (20.4%) | identical |
| Llama FLAT (45 ungrounded) | 24 (53.3%) | 0 (0.0%) | 21 (46.7%) | 12 (25.5%) user, 0 policy, 35 (74.5%) invented |
| Qwen CONDITIONAL (14 ungrounded) | 14 (100.0%) | 0 (0.0%) | 0 (0.0%) | identical |
| Llama CONDITIONAL (43 ungrounded) | 22 (51.2%) | 0 (0.0%) | 21 (48.8%) | 10 (22.2%) user, 0 policy, 35 (77.8%) invented |

**Unverified-user-ID error rate: confirmed 38/43 (88.4%).** This is A8's
strict definition (matched tasks 0/1/3, both agents, calls with >=1
ungrounded id of `origin == user_utterance`, numerator = that call
errored) under the component-level classifier - `paper_numbers.py`'s
`strict_user_stated_call_error_rate(component_level=True)`. As-submitted:
32/37 (86.5%).

**Mismatched vs matched gap and the artifact share:**

| | component-level | as-submitted |
|---|---|---|
| mismatched gap (Qwen self-play vs Llama-on-1.5B-user) | 77.2pp | 48.3pp |
| matched gap (T1, repeated) | 48.1pp | 34.9pp |
| **fraction of gap attributable to simulator artifact** | **37.7%** | 27.8% |

## T4 - exposure (definition ii), agent held fixed

**Llama x 7B user** (matched, `llama_task{0,1,3}.json`):

| task | n/d (%) | as-submitted |
|---|---|---|
| task0 | 9/10 (90.0%) | 5/8 (62.5%) |
| task1 | 5/10 (50.0%) | 5/12 (41.7%) |
| task3 | 2/7 (28.6%) | 2/8 (25.0%) |
| **pooled** | **16/27 (59.3%)** | 12/28 (42.9%) |

**Llama x 1.5B user** (`llama_task{0,1,3}_1.5b_user.json`):

| task | n/d (%) | as-submitted | note |
|---|---|---|---|
| task0 | 3/14 (21.4%) | identical | **20-step smoke test** |
| task1 | 3/16 (18.8%) | identical | |
| task3 | 0/11 (0.0%) | 0/9 (0.0%) | |
| **pooled** | **6/41 (14.6%)** | 6/39 (15.4%) | |

**Qwen x 7B user, tasks 0/1/3 ONLY (task2 excluded)**:

| task | n/d (%) | as-submitted |
|---|---|---|
| task0 | 5/11 (45.5%) | identical |
| task1 | 8/9 (88.9%) | identical |
| task3 | 4/7 (57.1%) | identical |
| **pooled** | **17/27 (63.0%)** | identical |

Reading: holding the agent fixed at Llama isolates a real, large
user-model effect (7B 59.3% vs 1.5B 14.6%, component-level). A residual
agent effect also survives at the fixed 7B user model (Qwen 63.0% vs
Llama 59.3%) - smaller than the raw T4 contrast, but not zero.

## T5 - schema conformance (confirmed unchanged)

| | component-level | as-submitted |
|---|---|---|
| Qwen malformation | 0/8 (0.0%) | identical |
| Llama malformation | 10/10 (100.0%) | identical |
| parallel-call rejection | 2/2 | identical |

Provably unaffected: `schema_conformance.py` has zero import dependency
on `failure_analysis.py`/`component_level_grounding.py`; its
`classify_arg()` operates on raw Python types only. Re-run this session
anyway to confirm: exact match.

## Section 3.5 - post-error distribution (confirmed unchanged)

| next action | count | share |
|---|---|---|
| lookup/verification tool | 29 | 51.8% |
| same tool, different args | 10 | 17.9% |
| exact same call repeated | 4 | 7.1% |
| something else | 9 | 16.1% |
| no next call | 4 | 7.1% |
| **n** | **56** | |

Identical under both classifiers, provably: `next_action_counts` is
built only from raw call name/arguments and error flags, never from
`id_values`/`classify_origin`/`is_grounded_in`. No table number in the
paper (§3.5 is prose + an unlabeled table there too, separate from the
T1-T5 scheme).

## Denominator audit - widening to email/zip/name fields

Pooled conditional ungrounded rate (both agents, matched tasks 0/1/3,
component-level), comparing the current denominator (NARROW `_id`/`_ids`
suffix fields PLUS the WIDE fields `email`/`zip`/`first_name`/`last_name`)
against a NARROW-only denominator (suffix fields alone):

| definition | n/d | rate |
|---|---|---|
| WIDE (current) | 57/110 | 51.8% |
| NARROW only | 47/98 | 48.0% |
| **change from widening** | - | **+3.9pp** |

Widening the denominator to include `email`/`zip`/`first_name`/
`last_name` raises the pooled conditional ungrounded rate by 3.9
percentage points (both agents pooled) - a real but modest effect of
that design choice, not large enough to be a confound by itself, but
worth disclosing since it IS a choice, not a forced consequence of the
tool schemas (the retail domain's tools don't require these fields to be
`_id`-suffixed; `WIDE_FIELD_NAMES` is a judgment call documented in
`failure_analysis.py`'s own comments).

## Hand-audit: component-level "invented" IDs not already covered by the 46-ID audit

The original 46-ID audit (`docs/PROVENANCE.md`) covered every
as-submitted invented value, tasks 0/1/3, both models. Component-level
parsing decomposes some of those (the stringified lists) into different,
smaller pieces - some pieces are no longer invented (grounded, or
reclassified `user_utterance`), but **6 pieces are still invented under
component-level parsing and were never individually audited before**
(only their MERGED, pre-decomposition form was):

| conversation | turn | field | value | causal user-text prefix | verdict |
|---|---|---|---|---|---|
| llama_task0 | 4 | new_item_ids | `1151293681` | 1 turn, no mention | **true invention** |
| llama_task0 | 4 | new_item_ids | `4983901481` | 1 turn, no mention | **true invention** |
| llama_task0 | 12 | new_item_ids | `1151293681` | 3 turns, only the OLD ids (1151293680/4983901480) stated | **true invention** |
| llama_task0 | 12 | new_item_ids | `4983901481` | 3 turns, only the OLD ids stated | **true invention** |
| llama_task1 | 16 | new_item_ids | `1151293681` | 2 turns, only model codes (W2378156-KB etc.), no digit IDs | **true invention** |
| llama_task1 | 16 | new_item_ids | `4983901481` | 2 turns, same | **true invention** |

Checked against the full causal user-text prefix (every user turn strictly
before that call), with exact substring, case/whitespace/quote-normalized,
and `#`-stripped variants, same methodology as `docs/PROVENANCE.md`.
**Agreement: 6/6. Disagreements: 0.** Each of these 6 is the FIRST
causal occurrence of its value in its conversation (the agent proposes a
specific replacement item ID before the user has stated it); in every
case the same digit value is later typed by the user and correctly
reclassified `user_utterance` at a subsequent turn (turn16/22/28/50 for
`llama_task0`, turn26/40 for `llama_task1`) - the same "agent proposes
first, user confirms later" pattern already documented, now visible at
the component level too. No new disagreement, no number changed.

The remaining 15 component-level-invented instances (`llama_task0`:
`payment_method_id` x6 + `email` x1; `llama_task1`: `email`/`first_name`/
`last_name` x1 each; `llama_task3`: `order_id`/`payment_method_id`/
`first_name`/`last_name`/`zip` x1 each) are identical, unaffected
non-list-field values already covered by the original 46-ID audit - not
re-audited here, nothing about them changed.

## Not recomputed / out of scope

- `paper/PAPER.md`, `ANALYSIS_PLAN.md`, `ANALYSIS_PLAN_AMENDMENT.md` (not
  edited).
- Part B (dropped, see `PROGRESS.md`).
