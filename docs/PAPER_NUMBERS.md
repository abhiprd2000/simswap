# Every paper number, as-submitted vs component-level

**Superseded by `docs/CAMERA_READY_NUMBERS.md`** (written 2026-10-18,
corrected 2026-10-24) - that document is the final, authoritative
claim-by-claim table; this one is kept for the A7-A10 narrative but any
number here that disagrees with `CAMERA_READY_NUMBERS.md` is wrong here,
not there (e.g. the permutation p-value below was fixed from 0.111 to
0.125 - see `CAMERA_READY_NUMBERS.md`'s "Corrected 2026-10-24" note).

Written 2026-10-12, updated 2026-10-18 (A8/A9/A10). Component-level parsing
(`src/component_level_grounding.py`) is the paper's PRIMARY classifier per
the operator's instruction (A7): a stringified list is a schema failure
already counted on the conformance axis; scoring it again as an
invented/ungrounded ID counts one behavior twice. As-submitted (default
`failure_analysis.py`) numbers are kept for disclosure, not deleted.
Source of truth: `src/paper_numbers.py`, writes `paper_numbers_raw.json`.
Nothing in `failure_analysis.py`, `schema_conformance.py`,
`matched_comparison.py`, `permutation_test.py`, or `user_id_mentions.py`
was modified.

## Numbering used in this document (A10)

T1-T5 below is a **relabeling requested for this document, not the
numbering currently live in `paper/PAPER.md`**. Three different
numberings now exist for the same content - stated plainly so nobody
conflates them:

| content | paper/PAPER.md's current label | README.md's inline-comment label | this document (A10) |
|---|---|---|---|
| grounding/conditional rate (§3.1) | Table 1 | Table 1 | **T1** |
| per-task breakdown under §3.1 | unlabeled (no table number) | - | **T2** |
| provenance of ungrounded IDs (§3.2) | Table 2 | Table 3 | **T3** |
| schema conformance (§3.3) | Table 3 | Table 2 | **T5** |
| mechanism / exposure (§3.4) | Table 4 | Table 4 | **T4** |
| post-error next action (§3.5) | Table 5 | Table 5 | **no table, Section 3.5** |

This document uses the **A10 column** throughout. `paper/PAPER.md` itself
has not been touched - if this renumbering is adopted, the paper's own
table labels and the README's inline comments both need updating to
match (not done here, out of scope for a numbers-only response).

## T1 - conditional ungrounded rate, both agents, n/d

| claim | as-submitted | component-level | source |
|---|---|---|---|
| Qwen | 14/15 (93.3%) | 14/15 (93.3%) - identical, zero stringified lists | `paper_numbers.py` |
| Llama | 45/77 (58.4%) | 43/95 (45.3%) | `paper_numbers.py` |
| **gap** | 34.9pp | **48.1pp** | `paper_numbers.py` |

## T2 - per-task conditional rates and gaps, tasks 0/1/3

| task | Qwen as-sub. | Qwen comp. | Llama as-sub. | Llama comp. |
|---|---|---|---|---|
| task0 | n/d (pool never fills) | n/d | 22/38 (57.9%) | 22/50 (44.0%) |
| task1 | 6/6 (100.0%) | 6/6 (100.0%) | 15/29 (51.7%) | 15/35 (42.9%) |
| task3 | 8/9 (88.9%) | 8/9 (88.9%) | 8/10 (80.0%) | 6/10 (60.0%) |

Permutation test (per-task FLAT diffs, Qwen-Llama), exact enumeration of
all 2^3=8 sign assignments:

| | as-submitted | component-level |
|---|---|---|
| per-task diffs | +0.410, +0.467, +0.137 | +0.549, +0.556, +0.338 |
| all 3 same sign | yes (Qwen higher) | yes (Qwen higher) |
| one-sided p | 0.125 (corrected 2026-10-24, was 0.111) | **0.125 - unchanged** |
| two-sided p | 0.25 (corrected 2026-10-24, was 0.222) | **0.25 - unchanged** |

## T3 - provenance of ungrounded IDs, flat AND conditional, counts and %

| | as-submitted | component-level |
|---|---|---|
| Qwen FLAT (54 ungrounded) | user 43 (79.6%), invented 11 (20.4%) | identical - no lists |
| Llama FLAT (47 / 45 ungrounded) | user 12 (25.5%), invented 35 (74.5%) | user 24 (53.3%), invented 21 (46.7%) |
| Qwen CONDITIONAL (14 ungrounded) | user 14 (100.0%), invented 0 (0.0%) | identical |
| Llama CONDITIONAL (35 / 21 ungrounded) | user 10 (22.2%), invented 35 (77.8%) | user 22 (51.2%), invented 21 (48.8%) |

## T4 - exposure (mechanism), per task, both user models, smoke test marked

Definition (ii) only - see `docs/TABLE4_METRIC_DEFINITIONS.md` for why (i)
isn't computable exactly.

| conversation | as-submitted | component-level | note |
|---|---|---|---|
| qwen_task0 | 5/11 (45.5%) | identical | |
| qwen_task1 | 8/9 (88.9%) | identical | |
| qwen_task2 | 7/13 (53.8%) | identical | |
| qwen_task3 | 4/7 (57.1%) | identical | |
| llama_task0 (1.5B) | 3/14 (21.4%) | identical | **20-step smoke test** |
| llama_task1 (1.5B) | 3/16 (18.8%) | identical | |
| llama_task3 (1.5B) | 0/9 (0.0%) | **0/11 (0.0%)** - denominator grows (two 2-element lists decompose), numerator stays 0 | |

### A9 - does T4 hold the agent fixed? No.

From `user_id_mentions.SOURCES`/`USER_MODEL_BY_CONVO`: `qwen_task0-3` all
read `data/raw/qwen_task{N}.json` (**agent = Qwen** throughout, user =
Qwen2.5-7B-AWQ); `llama_task0/1/3` all read
`data/raw/llama_task{N}_1.5b_user.json` (**agent = Llama** throughout,
user = Qwen2.5-1.5B-fp16). **The "7B row" and "1.5B row" differ in BOTH
agent and user model simultaneously - T4 does not hold the agent fixed.**
Confirmed, not assumed.

Agent-fixed recompute - Llama x 7B user (matched, `llama_task{0,1,3}.json`)
vs Llama x 1.5B user (`llama_task{0,1,3}_1.5b_user.json`), definition (ii):

| task | Llama x 7B as-sub. | Llama x 7B comp. | Llama x 1.5B as-sub. | Llama x 1.5B comp. |
|---|---|---|---|---|
| task0 | 5/8 (62.5%) | 9/10 (90.0%) | 3/14 (21.4%) | 3/14 (21.4%) |
| task1 | 5/12 (41.7%) | 5/10 (50.0%) | 3/16 (18.8%) | 3/16 (18.8%) |
| task3 | 2/8 (25.0%) | 2/7 (28.6%) | 0/9 (0.0%) | 0/11 (0.0%) |
| **pooled** | **12/28 (42.9%)** | **16/27 (59.3%)** | **6/39 (15.4%)** | **6/41 (14.6%)** |

Qwen x 7B user, matched tasks 0/1/3, reported separately (**no Qwen x
1.5B data exists** - Qwen was never run against the 1.5B user simulator):

| task | as-submitted | component-level |
|---|---|---|
| task0 | 5/11 (45.5%) | identical |
| task1 | 8/9 (88.9%) | identical |
| task3 | 4/7 (57.1%) | identical |
| **pooled** | **17/27 (63.0%)** | **identical** |

**Reading this**: holding the agent fixed at Llama, the user-model effect
is large and in the expected direction (7B user: 42.9-59.3% exposure vs
1.5B user: 14.6-15.4%) - a real user-simulator effect, independent of
agent. But a residual AGENT effect also survives even with the user model
held fixed at 7B: Qwen x 7B (63.0%) is still higher than Llama x 7B
(42.9% as-submitted, 59.3% component-level). T4 as currently constructed
cannot distinguish these two effects because it never holds the agent
fixed - this breakdown does.

## "86.5%" (A8) - reproduced and scripted

**Not computed by any repo script before this session** (confirmed:
`grep` for "86.5" across the whole repository found zero hits prior to
A7). It was computed in chat before submission. Added to
`paper_numbers.py` as `strict_user_stated_call_error_rate()`.

**Exact definition**: population = matched tasks 0/1/3, both agents, 7B
user (6 conversations: `qwen_task0/1/3` + `llama_task0/1/3` matched). A
call counts in the denominator if >=1 of its id-arguments is **ungrounded
AND has `origin == "user_utterance"`** (not merely "value appears
somewhere in user text", which is broader - see below). Numerator: those
calls whose own tool result errored.

| population | as-submitted | component-level |
|---|---|---|
| matched tasks 0/1/3, both agents | **32/37 (86.5%)** - matches exactly | 38/43 (88.4%) |
| all 7 conversations (incl. qwen_task2) | **36/42 (85.7%)** - matches exactly | 42/48 (87.5%) |

Per-conversation (default classifier, matched set): qwen_task0 5/5
(100.0%), qwen_task1 10/11 (90.9%), qwen_task3 9/10 (90.0%), llama_task0
1/3 (33.3%), llama_task1 6/7 (85.7%), llama_task3 1/1 (100.0%).

### Explaining the earlier (A7) 41/61

A7's draft used a broader, over-inclusive check: "value appears ANYWHERE
in prior user text", regardless of grounded/origin status - so it also
counted calls whose id was already GROUNDED and separately happened to
be user-stated too (e.g. a verified order ID that the user also
mentioned). Rerunning that broader check against the same population
reproduces 41/61 (67.2%) exactly. The 24 extra calls it includes over the
strict 32/37, listed and checked one by one
(`paper_numbers_raw.json`'s `a8_extra_calls_broad_minus_strict`):
**every single one of the 24 has >=1 id that is GROUNDED** (confirmed by
code, `all(...) == True`) - exactly the expected explanation. Mostly
repeated `get_order_details(order_id='#W2378156')` calls (already
grounded from an earlier successful lookup) and `exchange_delivered_order_items`
calls whose `order_id` was grounded even though their `item_ids`/
`new_item_ids` were separately invented. **Confirmed, not refuted.**

## Mismatched-simulator comparison (Qwen self-play vs Llama on the 1.5B user)

| | as-submitted | component-level |
|---|---|---|
| Qwen (self-play) conditional | 14/15 (93.3%) | identical |
| Llama-on-1.5B-user conditional | 27/60 (45.0%) | **10/62 (16.1%)** |
| mismatched gap | 48.3pp | **77.2pp** |
| matched gap (repeated from T1) | 34.9pp | 48.1pp |
| fraction of mismatched gap attributable to simulator mismatch | (48.3-34.9)/48.3 = **27.8%** | (77.2-48.1)/77.2 = **37.7%** |

The 1.5B-user Llama transcripts also contain 17 stringified-list id-args
(checked, not assumed, via `find_all_stringified_list_values`) - all 17
turn out fully grounded once decomposed (hand-traced on `llama_task1`'s
13 occurrences; arithmetic closes exactly: 27 old ungrounded - 17
recovered = 10 new). **A larger share of the original 48.3pp gap is
attributable to the user-simulator mismatch under component-level
parsing, not a smaller one.**

## T5 - schema conformance (expected unchanged - confirmed)

| | as-submitted | component-level |
|---|---|---|
| Qwen malformation | 0/8 (0.0%) | **identical - provably unaffected** |
| Llama malformation | 10/10 (100.0%) | **identical** |
| parallel-call rejection | 2/2 | **identical** |

`schema_conformance.py` imports nothing from `failure_analysis.py` and
calls `classify_arg()` directly on each argument's raw Python type (`list`
vs `str`) - zero dependency on `id_values`/`classify_origin`/
`is_grounded_in`. Re-run this session to confirm anyway: exact match.

## Section 3.5 - post-error distribution, all categories (expected unchanged - confirmed)

| next action | count | share |
|---|---|---|
| lookup/verification tool | 29 | 51.8% |
| same tool, different args | 10 | 17.9% |
| exact same call repeated | 4 | 7.1% |
| something else | 9 | 16.1% |
| no next call | 4 | 7.1% |
| **n** | **56** | |

**Identical under both classifiers - provably, not just empirically.**
`next_action_counts` is built entirely from `flat_calls` (`{name, RAW
arguments}`, exactly as submitted) and `error_flags` - this code path
never calls `id_values()`/`classify_origin()`/`is_grounded_in()`.
Component-level parsing only changes how id VALUES inside an
already-fixed call are graded; it cannot change which calls exist, their
names, their raw arguments, or which errored - the only inputs this table
uses. Not rerun, because there is nothing a rerun could find.

## What changes, and the two findings that most affect paper text

**1. T1's gap WIDENS under component-level parsing, it does not narrow:
34.9pp -> 48.1pp.** Counter to the naive "fixing double-counting should
shrink the gap" expectation. Qwen's rate is unaffected (no lists); Llama's
drops substantially (58.4% -> 45.3%, malformed `item_ids` turn out to be
correctly grounded once decomposed). A rate dropping on one side only
widens the gap. T3's provenance SPLIT still narrows sharply (77.8% ->
48.8% invented) even though T1's RATE gap widens - these are different
claims and move in different directions; neither should stand in for the
other.

**2. The "fraction of gap that was simulator artifact" sentence should
use 37.7%, not 27.8%,** if added under the new classifier - see above.

**3 (new, A9). T4 conflates agent and user-model effects.** Holding the
agent fixed at Llama isolates a real, large user-model effect (7B:
42.9-59.3% vs 1.5B: 14.6-15.4%) - but a smaller residual agent effect
survives even at a fixed (7B) user model (Qwen 63.0% vs Llama 42.9-59.3%).
T4 as published cannot show this distinction.

## Not recomputed / out of scope

- `paper/PAPER.md` and `ANALYSIS_PLAN.md` (not edited, per instruction).
- The README/paper table-numbering inconsistency (flagged, not fixed).
