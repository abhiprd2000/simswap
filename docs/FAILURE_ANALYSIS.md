# Failure analysis — ungrounded tool-call IDs

Written 2026-08-16, on operator instruction after the rank-inversion
design was stopped (0/4 sampled episodes ran to a normal completion,
so there is no score spread to invert a ranking over — see
`DECISIONS_NEEDED.md` #0 and #2's update). This is a characterization
of the failure mode itself, computed from runs already on disk plus
two rounds of targeted new GPU runs (Llama on additional tasks, see
below) — not a wider sweep.

**REVISED 2026-08-16, four times on methodology, twice on data, plus
one independent re-verification.** Methodology history, in order: (1)
the grounding pool was silently incomplete in 2 of 5 conversations, (2)
4 identifier arguments (`email`, `zip`, `first_name`, `last_name`) were
missing from the denominator, (3) the token-matching mechanism itself
had two edge cases (a stripped "#" prefix, a 4-character minimum) that
under-counted real grounding — found while writing
`REGRESSION_TESTS.md`'s test suite, fixed by switching to direct
substring containment, by far the largest correction, (4) **the flat
rate itself was flagged as misleading, not wrong**: `qwen_task0`'s
16/16 (100%) is *arithmetically forced* by a grounding pool that was
EMPTY for the entire conversation (zero successful tool results ever
occurred) — that is not evidence the agent ignored available
information, because there was no information available to ignore.
Every number in this document now has a **CONDITIONAL** counterpart
(restricted to calls made after the pool was already non-empty, i.e.
after the agent actually had something it could have checked) sitting
alongside the flat one — the conditional number is the honest headline
throughout, not the flat one. Full history in `AUDIT_REPORT.md`.
**Numbers below are final**: test-covered (`test_failure_analysis.py`,
11/11 passing, four new cases added across two rounds — see
`REGRESSION_TESTS.md`) and now include a third Llama task (task3),
close to matching Qwen's 4. The tasks-0+1 hand-verification (Q1) was
independently redone by a separate agent instance, tracing both raw
JSON transcripts directly, arriving at the identical 16/16 + 23/23 =
39/39 with no shortcuts - two independent hand-traces now agree, not
one.

## Method and definitions

Source transcripts (`sim_dump.messages` inside each `raw_tau2_dump_*.json`,
all retail domain, all real GPU runs, tau2-bench v1.0.1):

| label | model (agent side) | task | file |
|---|---|---|---|
| qwen_task0 | Qwen2.5-7B-Instruct-AWQ | 0 | `kaggle_output_passb_awq/raw_tau2_dump_passB_awq.json` |
| qwen_task1 | Qwen2.5-7B-Instruct-AWQ | 1 | `kaggle_output_passb_multitask/raw_tau2_dump_passB_awq_task1.json` |
| qwen_task2 | Qwen2.5-7B-Instruct-AWQ | 2 | `kaggle_output_passb_multitask/raw_tau2_dump_passB_awq_task2.json` |
| qwen_task3 | Qwen2.5-7B-Instruct-AWQ | 3 | `kaggle_output_passb_multitask/raw_tau2_dump_passB_awq_task3.json` |
| llama_task0 | Llama-3.1-8B-Instruct-AWQ | 0 | `kaggle_output_smoke_attempt2/raw_tau2_dump_smoke_llama31_8b_awq.json` |
| llama_task1 | Llama-3.1-8B-Instruct-AWQ | 1 | `kaggle_output_llama_multitask/raw_tau2_dump_llama_task1.json` |
| llama_task3 | Llama-3.1-8B-Instruct-AWQ | 3 | `kaggle_output_llama_task23/raw_tau2_dump_llama_task3.json` |

**Llama data gap, now 3 of 4:** task2 was attempted twice (real GPU
runs, independent attempts) and failed identically both times: vLLM's
`llama3_json` tool-call parser hard-rejects a genuine parallel
tool-call attempt ("This model only supports single tool-calls at
once!") — real, reproducible model/parser behavior, not analyzable as
a transcript, not retried further. task3 initially failed the same way
task2 did structurally (a crash after the conversation completed, no
transcript saved) but for a different, fixed reason: tau2's
NL-assertion evaluator needs a real LLM (OpenAI, not the local vLLM
endpoints) and no `OPENAI_API_KEY` was configured. A crash-guard patch
(`gate1_core.py`, verified against real tau2 source) now catches this
and returns a stub reward instead of losing the transcript — task3's
real conversation (26 turns, `termination_reason: normal`, the first
genuinely clean completion seen anywhere in this campaign) was
recovered this way. Its `reward`/`task_success` fields are stub
placeholders, clearly marked via `reward_info_note` — only the reward
judgment is fake, the transcript is real and used normally here.

## "Ungrounded" definition

The operator's phrasing — "never appeared in a prior tool RESULT" —
is ambiguous about error echoes, made explicit and auditable, and
revised twice already (see history above): an ID-shaped argument —
any argument whose field name ends in `_id`/`_ids` (`order_id`,
`item_id`, `item_ids`, `product_id`, `user_id`, `payment_method_id`,
etc.) **or** is one of `email`/`zip`/`first_name`/`last_name` (the
retail tool schema's other identifying arguments, confirmed against
tau2-bench source — see `AUDIT_REPORT.md` AUDIT 1) — is **grounded**
if its exact string value appears as a **substring** of the content of
an earlier **successful** (`error` false/absent) tool result in the
same conversation, checked causally (prior calls only — see
`AUDIT_REPORT.md` AUDIT 2 for how much the causal-vs-anywhere and
error-inclusion choices move the number on their own: 1-6 points,
secondary to the matching-mechanism fix). Error-result content is
excluded from the grounding pool on purpose: tau2's error messages
usually echo the bad ID back ("Error: Order #W123 not found"), which
would trivially and misleadingly count as "grounding" if included.

**FLAT vs CONDITIONAL, and why both are reported**: the FLAT rate (the
only one this document originally reported) is ungrounded ID arguments
divided by ALL ID arguments in a conversation — including calls made
while the grounding pool was still empty, when grounding was not yet
*possible*. An empty-pool call is ungrounded by construction, not by
choice. The CONDITIONAL rate restricts the same computation to only
the ID arguments/calls issued **after** the pool already held at least
one successful result — i.e. after the agent genuinely had something
it could have checked against. This is computed per call (all IDs
within one assistant turn share that turn's pool state, tracked as
`pool_nonempty` in `tool_calls_log`) and aggregated per conversation as
`*_conditional` fields, plus a `cascade` block per conversation:
`pool_ever_nonempty` (did any success ever occur at all), whether the
first ID-bearing call was grounded/errored, and how many calls happened
before the first success. A conversation whose pool never fills (e.g.
`qwen_task0`) contributes **0/0 — no conditional evidence at all**, not
a misleading 0% or 100%.

Full per-conversation and per-tool-call detail:
`failure_analysis_raw.json`. Scripts: `failure_analysis.py` (main
metric + origin classification + conditional/cascade tracking,
test-covered — `test_failure_analysis.py`),
`failure_analysis_next_action.py` (precise next-action-after-error,
see Q4, independent of the grounding-pool logic above and unaffected
by any of the fixes), `matched_comparison.py` (Q2's matched number,
aggregates `failure_analysis_raw.json` over the task-0/1/3 subset only
INCLUDING the conditional fields, writes `matched_comparison_raw.json`,
does not touch `failure_analysis.py`'s logic).

## Q1 — headline number: what fraction of tool calls use an ungrounded ID?

**FLAT, Qwen2.5-7B-Instruct-AWQ, pooled across all 4 sampled tasks
(reported for completeness, NOT the headline — see conditional below):**

- By ID argument: **69 / 86 = 80.2%** of individual ID-shaped
  arguments were never grounded in a prior successful tool result.
- By tool call: **38 / 43 = 88.4%** of tool calls that carried at
  least one ID argument carried at least one ungrounded one.

Per-task breakdown (by argument): task0 16/16 (100%), task1 23/23
(100%), task2 15/31 (48.4% — the task with the most successful
lookups), task3 15/16 (93.8%).

**CONDITIONAL (the honest headline — see the note above on flat vs
conditional):** restricting to ID arguments issued after the pool
already held >=1 successful result removes `task0` entirely (0/0, no
conditional evidence — its pool never fills, see the cascade note
below) and changes task1/task3's denominators:

| task | conditional | flat (for comparison) |
|---|---|---|
| task0 | **NO DATA** — pool never non-empty | 16/16 (100%) |
| task1 | 6/6 (100%) | 23/23 (100%) |
| task2 | 12/28 (42.9%) | 15/31 (48.4%) |
| task3 | 8/9 (88.9%) | 15/16 (93.8%) |
| **pooled (task1+2+3 only — task0 contributes nothing)** | **26/43 = 60.5%** | 69/86 (80.2%) |

**task1's conditional number (6/6, 100%) is the single most damning
data point in this whole analysis, not a weaker one.** It means: after
Qwen obtained the ONE real, verified piece of information in that
entire conversation (`find_user_id_by_name_zip` → `"yusuf_rossi_9620"`,
see the hand-trace below), it went on to make 6 more tool calls and
grounded *none* of them — not even by accident. This is the cleanest
possible case of "had it, didn't use it," free of any empty-pool
confound.

### Is this a cascade from a bad entry point, or a flat failure rate?

Hypothesis tested (per instruction): the first ID-bearing call is
ungrounded by construction (pool starts empty), so if it also errors,
the pool never fills and everything after is forced-ungrounded by
arithmetic — a cascade, not a flat rate. Per conversation, in call
order:

| conversation | first call ungrounded | first call errored | calls before first success | pool ever fills? |
|---|---|---|---|---|
| qwen_task0 | yes | **yes** | 11 (all of them) | **NO — cascade total collapse** |
| qwen_task1 | yes | **yes** | 5 | yes, at call 5 of 11 |
| qwen_task2 | yes | no | 1 | yes, immediately |
| qwen_task3 | yes | **yes** | 3 | yes, at call 3/4 |
| llama_task0 | yes | no | 1 | yes, immediately |
| llama_task1 | yes | no | 1 | yes, immediately |
| llama_task3 | no* | no | 1 | yes, immediately |

*llama_task3's first TOOL call overall had no ID arguments; the first
**ID-bearing** call came after the pool was already non-empty, hence
"no" here.

**The cascade hypothesis holds cleanly for exactly one conversation
(`qwen_task0`, total collapse — pool never fills, 11/11 calls happen
in an empty pool, 0 conditional evidence) and partially for two more
(`qwen_task1`, `qwen_task3` — first call errors, delaying but not
preventing eventual success). It does NOT hold for the other four**
(`qwen_task2`, all three Llama tasks) — their very first tool call
succeeds, the pool fills immediately, and every subsequent grounding
failure in those conversations is a genuine "had it, didn't use it"
case, not empty-pool arithmetic. **The honest reading: `qwen_task0`'s
100% is a cascade artifact and should not be quoted as a rate at all
(no conditional evidence exists); the other six conversations'
ungrounded calls, flat or conditional, are real behavioral signal.**
Computed by `analyze_conversation`'s `cascade` field, per-conversation
detail in `failure_analysis_raw.json`, locked down by
`test_empty_pool_conversation_has_no_conditional_evidence` and
`test_conditional_rate_excludes_only_pre_success_calls`
(`REGRESSION_TESTS.md`).

**Tasks 0 and 1's underlying call-by-call detail (not the flat 39/39
headline, which the cascade analysis above already qualifies) is
hand-verified**, not just script output — the operator's request,
given two prior bugs were both caught this way. **Independently redone
a second time**
(separate pass, same method, no shortcuts): task0 has 11 tool calls,
16 ID-shaped argument-values total (`order_id`×1, `item_ids`×2,
`new_item_ids`×2, `payment_method_id`×1 on the first
`exchange_delivered_order_items` call, plus 10 more single-`item_id`/
`product_id` calls) across 11 tool results, and **every single one of
the 11 results is `error: true`** — verified by reading every
`"role": "tool"` entry directly, not by trusting a count. Zero
successful results anywhere means zero possible grounding, full stop —
unambiguous, no edge case to argue about.

task1 has 11 tool calls, 23 ID-shaped argument-values, and **exactly
one** successful result: `find_user_id_by_name_zip(first_name="Yusuf",
last_name="Rossi", zip="19122")` → `"yusuf_rossi_9620"` (turn 31, call
5 of 11). Traced every one of the 6 subsequent calls individually: all
six are `get_order_details(order_id="W2378156")`, verbatim repeats,
every one an error. The string `"yusuf_rossi_9620"` never appears as
an argument anywhere, before or after it was obtained — the agent
finds the user ID and then never uses it. No accidental substring
overlap between `"yusuf_rossi_9620"` and `"W2378156"` either, so this
isn't a near-miss — the two values share no characters worth noting.
**No arguable or borderline case found in either transcript, on either
pass.** Full trace in `REGRESSION_TESTS.md`, locked down by
`test_all_error_conversation_is_fully_ungrounded`.

See also `SCHEMA_CONFORMANCE.md` for a second, independent failure
axis found in the same 7 transcripts: does the agent send arguments in
the shape the tool expects at all (e.g. a real list vs. a string that
looks like one)? Qwen: 0/8 malformed. Llama: 9/9 malformed. Plus a
distinct sub-axis: does the whole request even reach tau2 at all?
Llama's task2 was rejected outright by vLLM's parser (a genuine
parallel tool-call attempt, "This model only supports single tool-calls
at once!"), 2/2 independent attempts, 0 for Qwen. Neither is folded
into the grounding numbers here - three separate questions about the
same transcripts.

## CONFOUND CHECK, 2026-08-17 - matched user simulator, does the provenance contrast survive?

**Everything above this point uses the ORIGINAL data, where Qwen and
Llama faced different user simulators** (Qwen: self-play, Qwen2.5-7B-
Instruct-AWQ both sides. Llama: a fixed Qwen2.5-1.5B-Instruct fp16
user, not Llama self-play). Mechanism check (`user_id_mentions.py`,
`user_id_mentions_raw.json`, done prior to this session) found this is
a real confound, not just a labeling difference: the 7B user states
~3x as many distinct ID values per conversation as the 1.5B user, and
46-89% of the 7B user's stated-ID vocabulary reaches a tool call vs.
0-21% for the 1.5B user. Since Qwen's dominant failure mode is
"trusts an ID the user stated," and the two models' users said
different numbers of IDs, part of the observed Qwen/Llama gap could be
a user-simulator artifact rather than an agent difference.

**Fix**: re-ran Llama (agent) against Qwen2.5-7B-Instruct-AWQ (user) -
the same user model Qwen already faces in self-play - on tasks 0/1/3,
full `max_steps=200` (task0 previously only had a 20-step smoke-test
sample; now matched to the same cap as the rest). Real GPU run via the
parameterized kernel (`kernels/gate1_run.py`, config in
`kernels/README.md`), output curated into `data/runs/llama_task{0,1,3}.json`
and `data/raw/llama_task{0,1,3}.json`. All 3 tasks landed real transcripts
(termination: task0 `normal` 74 turns, task1 `agent_error` 52 turns,
task3 `normal` 30 turns - task2 not attempted, same unfixable
`llama3_json` parallel-tool-call parser rejection as before, see Q2
below). Qwen's data is untouched (it was already self-play against
this same user). Recomputed with `matched_usersim_recompute.py`
(imports `failure_analysis.py`/`schema_conformance.py`'s functions
unchanged, new `SOURCES` dict pointed at the new Llama transcripts
only - the grounding/schema logic itself was not touched). Raw:
`matched_usersim_raw.json`, `matched_usersim_pooled.json`.

**Result: the contrast survives, direction unchanged, magnitude shrinks.**

| | Qwen (unchanged) | Llama, OLD (1.5B user) | Llama, NEW (matched 7B user) |
|---|---|---|---|
| FLAT matched (0,1,3) | 54/55 (98.2%) | 33/66 (50.0%) | **47/79 (59.5%)** |
| CONDITIONAL matched (headline) | 14/15 (93.3%) | 27/60 (45.0%) | **45/77 (58.4%)** |
| gap vs Qwen (conditional) | - | 48.3pp | **34.9pp** |
| origin of ungrounded IDs (flat, matched) | 79.6% user / 20.4% invented | 18.2% user / 81.8% invented | **25.5% user / 74.5% invented** |

Fixing the confound moves Llama's rate up (45.0% -> 58.4% conditional)
- consistent with the mechanism check's prediction (a user that states
more IDs gives the agent more unverified-but-correct claims it could
trust without checking, the same failure pattern Qwen shows) - but
Qwen's rate stays far higher (93.3% vs 58.4%), and the origin split
stays inverted in the same direction (Llama still invention-majority,
just less extremely: 74.5% vs the old 81.8%). **The central claim -
Qwen mostly trusts unverified user-stated IDs, Llama mostly invents -
is not an artifact of the user-simulator mismatch.** It weakens in
degree (the gap shrinks by about a quarter) but does not reverse or
disappear.

Schema conformance is unaffected in the qualitative sense (still
binary): Qwen 0/8, Llama now 10/10 malformed (was 9/9) - one more
list-bearing call surfaced by the longer/different conversations, same
100% rate. Post-error next-action pooled over the new matched set:
lookup-tool remains the plurality reaction (29/56 = 51.8%, vs the old
pooled ~48%).

**Paired permutation test (requested explicitly, exact enumeration,
`permutation_test.py`, `permutation_test_result.json`)**: paired by
task (n=3: tasks 0,1,3), statistic = per-task FLAT ungrounded-rate
difference (Qwen - Llama) - flat rate used, not conditional, because
Qwen's task0 conditional rate is undefined (0/0, cascade collapse, see
Q1) and an exact paired test needs a real number for every pair. All
3 tasks show the same direction (Qwen higher): +0.410, +0.467, +0.137.
Null: each task's sign is exchangeable, 2^3=8 possible sign
assignments, small enough to enumerate exactly (no Monte Carlo).
**One-sided p = 1/9 = 0.111, two-sided p = 2/9 = 0.222.** This is the
smallest possible p-value obtainable with only 3 paired tasks (1/9 is
the floor of this exact test) - a perfectly consistent 3-for-3 result
still cannot reach conventional significance (p<0.05) at this sample
size. Report the direction (consistent across all 3 matched tasks) and
the effect size (34.9pp conditional gap), not a significance claim -
none is supportable here, and none was found.

**What this means for the paper**: the headline contrast (rate and
provenance) survives a common user simulator and should be reported
using the NEW Llama numbers (58.4% conditional, 74.5%/25.5%
invented/user split) as the current, deconfounded figures - the old
numbers (45.0%, 81.8%/18.2%) are now superseded, not just a footnote
caveat. No significance test supports the gap at n=3 matched tasks;
say so plainly if a significance claim is needed rather than omitting
the test. The user-ID-statement-rate mechanism finding
(`user_id_mentions_raw.json`: 7B user states ~3x more IDs, 46-89% vs.
0-21% reach a tool call) belongs in a mechanism section regardless of
this outcome - it is real and explains PART of why the gap shrank, not
why it didn't disappear.

## Q2 — does Llama do it too?

**Yes, but at a consistently and substantially lower rate than Qwen —
by every version of this number, flat or conditional, pooled or
matched.**

**THE NUMBER FOR THE ABSTRACT — matched (tasks 0/1/3, both models),
CONDITIONAL (pool already non-empty at call time):**

| model | conditional matched | tasks contributing evidence |
|---|---|---|
| Qwen | **14 / 15 = 93.3%** | 2 of 3 (task0 contributes 0/0 — cascade collapse, see Q1) |
| Llama | **27 / 60 = 45.0%** | 3 of 3 |

**Gap: 48.3 points.** Computed by `matched_comparison.py`, which now
aggregates the `*_conditional` fields alongside the flat ones — see
`matched_comparison_raw.json` for full per-task detail. Locked down by
`test_matched_pool_aggregates_correctly` (`REGRESSION_TESTS.md`).

**This is not simply the flat matched number with the edges sanded
off — it survives the operator's correction almost unchanged in gap
size (48.3 vs 48.2 points), while both models' own numbers move down
somewhat (Qwen 98.2%→93.3%, Llama 50.0%→45.0%), because removing
empty-pool calls removes noise from both sides, not just Qwen's.** The
one asymmetry worth flagging honestly: Qwen's conditional evidence
comes from only 2 of its 3 matched tasks (task0 contributes nothing —
its pool never fills, see Q1's cascade table), while Llama's comes
from all 3. This is disclosed, not hidden — it doesn't change the
direction or rough size of the gap, but a reader deserves to know the
Qwen number rests on a smaller effective sample than Llama's.

**For reference, the two numbers this document previously led with,
now demoted to supporting detail:**

- Flat, unmatched (Qwen all 4 tasks, Llama all 3): 69/86 (80.2%) vs
  33/66 (50.0%) — 30.2-point gap.
- Flat, matched (tasks 0/1/3 both models): 54/55 (98.2%) vs 33/66
  (50.0%) — 48.2-point gap. This was reported as "the number for the
  abstract" in the previous revision of this document — **superseded**
  by the conditional matched number above per the operator's
  correction: 98.2% overstates Qwen's rate by including `task0`'s
  cascade-forced 16/16, which is not behavioral evidence.

Per-task detail (flat, for context): Qwen task0 16/16 (100%), task1
23/23 (100%), task2 15/31 (48.4% — Qwen's best task, excluded from the
matched set), task3 15/16 (93.8%). Llama task0 7/15 (46.7%), task1
18/42 (42.9%), task3 8/9 (88.9% — small-denominator outlier, 26-turn
conversation that terminated normally rather than getting capped by
errors).

**The gap (by every measure) is stable across three tasks, not two,
and task2's absence is real and reproducible** (confirmed on two
independent GPU attempts, see `PROGRESS.md`), not something more GPU
time would likely resolve.

## Q3 — where do the ungrounded IDs come from?

**Computed on the FLAT ungrounded set (not conditional)** — origin
(user-stated vs. invented vs. policy) is a question about EVERY
ungrounded ID's source, including `task0`'s cascade-forced ones, since
"where did this value come from" is meaningful even for a call issued
before grounding was possible. No conditional version of this
breakdown was computed — flag this if a future revision needs one.

**Qwen (69 ungrounded IDs, pooled across 4 tasks):** 57 (82.6%) were
copied verbatim from something the **user said** earlier in the
conversation, 12 (17.4%) were **invented** with no traceable source,
0 from the policy document.
- **Reading this**: most of Qwen's "ungrounded" IDs are not blind
  hallucination from nothing — they're the agent trusting an ID the
  customer stated (e.g. an order number) and acting on it directly,
  without ever calling a tool to confirm it's real. That's still a
  real grounding failure (the number could be wrong, and the agent
  has no way to know without checking), but it's a different, more
  specific failure than "the model made up a number" — it's "the
  model skips verification of user-supplied identifiers."

**Llama (33 ungrounded IDs, pooled across 3 tasks):** 27 (81.8%)
**invented**, 6 (18.2%) from the **user**, 0 from policy. Per-task:
task0 was 57.1%/42.9% invented/user, task1 was 83.3%/16.7%, task3 was
100%/0% (all 8 ungrounded IDs in task3 were invented, none traced to
the user). Consistently invention-majority across all three tasks —
the direction never flipped once across three independent samples.

**This is now a robust, three-task-supported contrast, close to
parity with Qwen's sample size**: Qwen is user-trust-majority (82.6%
user-sourced) and Llama is invention-majority (81.8% invented) — these
two numbers are now nearly mirror images of each other. Combined with
Q2, the fuller picture: Qwen fails more often (80.2% vs 50.0%) but its
failures are mostly "trusted an unverified claim," while Llama fails
less often but the large majority of its failures are "fabricated
from nothing."

## Q4 — what does the agent do right after an error?

Computed precisely per error event (not the coarser field-pooled
heuristic used for Q1-Q3, and entirely independent of the grounding
pool — unaffected by any of the three fixes above, now across 7
conversations including llama_task3): for each of the error results,
classify the very next tool call as: an exact repeat of the same call
(name + arguments identical), the same tool with different arguments,
a call to a genuine lookup/verification tool (`get_order_details`,
`get_user_details`, `find_user_id_by_email`, `find_user_id_by_name_zip`,
`list_all_product_types`), something else, or no next call
(conversation ended there).

| next action | count | share |
|---|---|---|
| calls a genuine lookup tool | 25 | ~48% |
| same tool, different (still unverified) arguments | 7 | ~13% |
| exact same call repeated verbatim | 6 | ~11% |
| something else | 8 | ~15% |
| no next call (episode ended) | 5 | ~9% |

**Reading this**: the plurality reaction to an error is still a
genuine lookup-tool call, not blind retry or invention, consistent
with the 2-task version of this table. Per-conversation breakdown in
`failure_analysis_raw.json`'s `post_error_actions` if a per-model split
is needed later.

## What this does and doesn't establish

This is a description of tool-call grounding behavior on 7
conversations (4 Qwen, 3 Llama), not a claim about a rate that would
hold at Gate-2 scale. The identifier-argument field set (`_id`/`_ids`
suffix + `email`/`zip`/`first_name`/`last_name`) is checked against
the full retail tool schema (`AUDIT_REPORT.md` AUDIT 1) — every
argument in every public retail tool is either covered or is a
non-identifying field (`reason`, `address1/2`, `city`, `state`,
`country`, `summary`, `expression`) not meaningful to check for
"grounding" in this sense. The grounding-match logic, the flat/
conditional/cascade tracking, schema-conformance checks, and the
matched-comparison aggregation are covered by 11 regression tests
(`REGRESSION_TESTS.md`, `test_failure_analysis.py`, 11/11 passing)
after two rounds of real bugs, one methodology correction (flat vs.
conditional rate), and two independent hand-verification passes (Q1) —
this doesn't guarantee no further issues, but the specific failure
modes found so far are now locked down against regression. Llama's
sample (3 tasks) is now close to Qwen's (4 tasks); the matched
comparison (Q2) removes even that asymmetry, and the conditional
version of that comparison additionally removes the empty-pool
confound that inflated Qwen's flat matched number. task2's gap is real
and not expected to close with more GPU time.
