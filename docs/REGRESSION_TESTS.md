# Regression tests for the grounding-check logic

Written 2026-08-16 on operator instruction: `failure_analysis.py`
produces every number in `FAILURE_ANALYSIS.md`, and by the time this
was written it had already needed two rounds of bug fixes (see
`AUDIT_REPORT.md`). It needed tests before anything else got built on
top of it. Test file: `test_failure_analysis.py`. Run with
`python test_failure_analysis.py` — plain asserts, no pytest
dependency, non-zero exit on any failure, matches this repo's existing
script style (`detectability.py`, `fake_runs.py`).

**Writing these tests found a third real bug**, on the first run — not
a hypothetical exercise. See "What the tests actually caught" below.

## The 5 cases (all in `CASES`, `test_failure_analysis.py`)

1. **`test_grounded_via_non_suffixed_arg_tool`** — the original bug's
   exact regression guard. `find_user_id_by_name_zip` has no
   `_id`-suffixed argument; its successful result must still ground a
   later `get_user_details(user_id=...)` call using that value. This
   must never regress — it's the case that made the first version of
   `FAILURE_ANALYSIS.md` wrong.
2. **`test_id_only_in_error_result_stays_ungrounded`** — an ID that
   only ever appears in error-result content (echoed back, e.g.
   "Error: item FAKE-999 not found") must not count as grounded.
   Confirms the error-exclusion design choice is actually enforced.
3. **`test_causal_ordering_future_result_does_not_ground_past_use`** —
   an ID used *before* any successful result ever confirms it must be
   ungrounded at the time of use, even if that same ID shows up in a
   *later* successful result. Confirms causal (prior-only) ordering,
   not whole-conversation membership.
4. **`test_legitimate_grounding_same_and_later_turns`** — a genuinely
   grounded ID must be recognized both immediately after its grounding
   result (adjacent turn) and much later (non-adjacent turn), and an
   unrelated argument in that same later call that was never grounded
   must still show up as ungrounded. This is the case that failed
   twice (see below) before the underlying matching bug was actually
   fixed.
5. **`test_non_id_suffixed_arguments_are_in_the_denominator`** —
   `email`/`zip`/`first_name`/`last_name` (the AUDIT 1 denominator
   fix) must be picked up as ID-shaped arguments, not silently
   skipped.
6. **`test_all_error_conversation_is_fully_ungrounded`** (added
   2026-08-16, after hand-verifying `qwen_task0`/`qwen_task1` call by
   call per the operator's request) — a conversation with zero
   successful tool results must show every ID argument as ungrounded,
   with no false-positive grounding from an empty pool. Locks in the
   hand-verified fact that Qwen's headline 100%-on-two-tasks number is
   real, not an artifact.
7. **`test_schema_conformance_detects_str_instead_of_list`** (added
   2026-08-16, `schema_conformance.py`, the second failure axis) — a
   string that merely *looks* like a list (`"[1151293680]"`, the real
   `llama_task0` case) must be flagged as malformed; a genuinely
   well-formed list must not be; and the reverse mismatch (a list where
   a plain `str` field is expected) must also be caught.
8. **`test_empty_pool_conversation_has_no_conditional_evidence`**
   (added 2026-08-16, the operator's correction — see "The conditional
   rate" below) — **THE MOST IMPORTANT TEST IN THIS FILE, must never
   regress.** A conversation whose grounding pool is NEVER non-empty
   (every tool result an error, the real `qwen_task0` shape) has a flat
   ungrounded rate of 100% - but that is arithmetically forced, not
   evidence. The CONDITIONAL fields (`total_id_arguments_conditional`,
   `ungrounded_fraction_by_argument_conditional`, etc.) must report
   0/0 and `None`, never a silent 0% or 100%, and the `cascade` block
   must show every call happened before any success.
9. **`test_conditional_rate_excludes_only_pre_success_calls`** (added
   2026-08-16, same correction) — a conversation where the pool fills
   PARTWAY through must restrict the conditional metrics to only the
   calls issued after that point, and the flat and conditional rates in
   the fixture must genuinely differ (0.75 flat vs 0.5 conditional) -
   otherwise the test wouldn't be checking anything real.
10. **`test_matched_pool_aggregates_correctly`** (added 2026-08-16,
    updated same day for the conditional fields, `matched_comparison.py`,
    the task-0/1/3 matched comparison) — pooling must sum raw counts and
    recompute the fraction from the sums, not average per-task fractions
    directly (averaging would misweight tasks with different
    denominators - the test's fixture is built so the two answers would
    visibly differ, 0.5 vs 0.625, if the bug were present); the
    conditional aggregation must also correctly exclude a conversation
    whose pool never fills (0/0) without corrupting the pooled fraction.
11. **`test_schema_conformance_records_llama_parallel_tool_call_rejection`**
    (added 2026-08-16, `schema_conformance.py`'s `NON_TRANSCRIPT_EVENTS`,
    the second schema-conformance sub-axis) — `llama_task2`'s
    parallel-tool-call rejection has no transcript to derive it from, so
    nothing else would catch it being silently deleted in a future edit;
    confirms it's present for llama and absent for qwen.

## The conditional rate (2026-08-16, operator correction)

`qwen_task0`'s reported 16/16 (100%) ungrounded was flagged as
misleading: the grounding pool was EMPTY the entire conversation (all
11 tool results are errors, zero successes), so every ID argument was
ungrounded by construction — there was nothing to ground against, not
evidence the agent ignored something it had. `failure_analysis.py`'s
`analyze_conversation` now tracks `pool_nonempty` per call (state
BEFORE that call) and `errored` per call, and reports CONDITIONAL
metrics (`*_conditional` fields) restricted to calls made after the
pool already held >=1 success, plus a `cascade` block
(`pool_ever_nonempty`, whether the first ID-bearing call was
grounded/errored, `calls_before_first_success`). This is purely
additive - the existing `is_grounded_in`/`id_values`/causal-ordering
logic (tests 1-6 above) is unchanged and still passes exactly as
before. `matched_comparison.py`'s `pool()` aggregates the new fields
too. Full result in `FAILURE_ANALYSIS.md` Q1/Q2 - the conditional
matched number (Qwen 93.3%, Llama 45.0%) is now the headline, not the
flat matched number (98.2%/50.0%) this document previously pointed to.

## What the tests actually caught

Test 4 failed on first run: `#O2` (order_id) used in a later turn,
after a successful `get_order_details` result had already revealed it,
was still marked ungrounded. Traced to `TOKEN_RE`
(`r"[A-Za-z0-9][A-Za-z0-9_#-]{3,}"`, later `r"#?[A-Za-z0-9][A-Za-z0-9_#-]{3,}"`):
token extraction required the first character to be alphanumeric
(stripping the "#" off real tau2 order IDs) *and* a 4-character
minimum length, either of which alone can make a real, legitimately
grounded ID fail to exact-match its own extracted token. Making the
"#" optional fixed the reproduction case in isolation but the test
still failed against a shorter fixture ID, which is what surfaced the
length-minimum edge case too. Root cause was the token-extraction
approach itself, not a fixable regex detail — replaced with direct
substring containment (`is_grounded_in`: does the value literally
appear anywhere inside a prior successful result's raw content), which
has no token boundary or length-minimum failure mode. All 5 tests pass
after that change.

**This was not a small correction.** The substring fix moved the
pooled numbers by double digits in two conversations: qwen_task2
(71.0% → 48.4% ungrounded) and llama_task0 (86.7% → 46.7% ungrounded)
— both conversations where an order_id got legitimately revealed by a
successful lookup partway through. `FAILURE_ANALYSIS.md` and
`AUDIT_REPORT.md` are updated with these numbers; the two-bug history
(narrow denominator, then a two-part token-matching bug) is disclosed
in full there, not just here.

## Hand-verification (2026-08-16, operator request, before trusting the headline)

Two prior bugs were both caught by hand-tracing, not by inspection of
the code. Before accepting Qwen's pooled tasks-0+1 number (39/39,
100% ungrounded) as the paper's headline, both transcripts were traced
call by call:
- `qwen_task0`: all 11 tool results are errors — zero successful
  results exist in the whole conversation, so nothing could possibly
  be grounded. Unambiguous.
- `qwen_task1`: exactly one successful result exists
  (`find_user_id_by_name_zip` → `"yusuf_rossi_9620"`), but that
  returned value is never used as an argument in any later call — the
  agent asks for `get_order_details(order_id='W2378156')` eight times
  and never once calls anything with the user_id it already has.
  Correctly ungrounded, not a bug.

**No arguable or borderline case was found in either transcript** —
every ungrounded call is either the first mention of that value, or a
value that never appears in any successful result anywhere in the
conversation. `test_all_error_conversation_is_fully_ungrounded` (case
6) locks down the first pattern as a permanent regression guard.

## Current status

```
python test_failure_analysis.py
PASS: test_grounded_via_non_suffixed_arg_tool
PASS: test_id_only_in_error_result_stays_ungrounded
PASS: test_causal_ordering_future_result_does_not_ground_past_use
PASS: test_legitimate_grounding_same_and_later_turns
PASS: test_non_id_suffixed_arguments_are_in_the_denominator
PASS: test_all_error_conversation_is_fully_ungrounded
PASS: test_schema_conformance_detects_str_instead_of_list
PASS: test_empty_pool_conversation_has_no_conditional_evidence
PASS: test_conditional_rate_excludes_only_pre_success_calls
PASS: test_matched_pool_aggregates_correctly
PASS: test_schema_conformance_records_llama_parallel_tool_call_rejection

11/11 passed
```

Re-run this after any change to `failure_analysis.py`'s grounding or
conditional/cascade logic (`is_grounded_in`, `id_values`, the
causal-ordering walk, or the `pool_nonempty`/`cascade` tracking in
`analyze_conversation`), `schema_conformance.py`'s type checks or
`NON_TRANSCRIPT_EVENTS`, or `matched_comparison.py`'s pooling
(including the conditional aggregation). A change that breaks any of
these 11 cases should not ship without understanding why.

## Second hand-verification pass (2026-08-16, independent re-check)

Before extending the analysis further (matched comparison, second
schema-conformance axis), the Q1 hand-verification above was redone
independently, from scratch, by tracing the raw JSON directly rather
than trusting the writeup: `qwen_task0` (11 tool calls, 16 ID
arguments, 11/11 tool results are errors, zero successes) and
`qwen_task1` (11 tool calls, 23 ID arguments, exactly 1 success -
`find_user_id_by_name_zip` -> `"yusuf_rossi_9620"` - whose value was
then traced through all 6 subsequent calls and confirmed never reused).
Same 39/39 result, same "no arguable case" conclusion, arrived at
independently. Two hand-traces agreeing is stronger evidence than one.
