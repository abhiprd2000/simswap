# Audits of FAILURE_ANALYSIS.md's methodology

Written 2026-08-16, on operator instruction, before any new GPU work.
Two audits requested: does the ID-argument heuristic miss real
identifier arguments (denominator), and how much does the grounding
rule itself move the headline number (robustness). **Doing this audit,
and then writing the regression tests requested alongside it, found
TWO real bugs in `failure_analysis.py` — not one — that made the first
two versions of `FAILURE_ANALYSIS.md` wrong, the second more than the
first. Both disclosed in full below, not smoothed over.**

## The bug found while building this audit

`failure_analysis.py`'s original grounding-pool update was nested
inside `if result_queue: ...` — it only recorded a successful tool
result's content as "grounding material" when the resolved call
itself had a tracked ID-shaped argument. A successful
`find_user_id_by_name_zip(first_name=..., last_name=..., zip=...)`
call has no `_id`/`_ids`-suffixed argument, so its result (a real
user_id, e.g. `"yusuf_rossi_9620"`) never entered the grounding pool
— even though a later `get_user_details(user_id="yusuf_rossi_9620")`
call using that exact, correctly-looked-up value should have counted
as grounded. It didn't. This under-counted grounding in exactly the
2 conversations (qwen_task2, qwen_task3) where the agent called
`find_user_id_by_name_zip`/`find_user_id_by_email` before using the
result — found by tracing `qwen_task2` call-by-call (`debug_diff.py`,
`debug_diff.log`) and confirming a hand-count against the code.
`audit_full.py`, written fresh for this audit, does not have this bug
(pool update is unconditional on any successful result). Fixed in
`failure_analysis.py`; both scripts now agree exactly. `qwen_task0`
and `qwen_task1` were unaffected (they don't call
`find_user_id_by_*` before the first use of the resulting ID), which
is why the bug wasn't visible in every conversation.

**Practical effect**: the original headline (89.0% Qwen ungrounded by
argument, 100% Qwen ungrounded by call) undercounted grounding. See
"Corrected numbers" below for the real ones.

## A second, bigger bug found while writing the regression tests

Building `REGRESSION_TESTS.md`'s test suite (operator instruction,
same session) surfaced a second, larger problem in the SAME script,
independent of the two fixes above. `TOKEN_RE`
(`r"[A-Za-z0-9][A-Za-z0-9_#-]{3,}"`) extracted candidate "grounded
tokens" from successful tool-result content, and grounding was tested
via exact membership of the argument value in that token set. Two
edge cases in that extraction both under-counted real grounding:

1. The regex required its first character to be `[A-Za-z0-9]`, so a
   leading `#` (every real tau2 order ID is `#W1234567`-shaped) got
   stripped before matching. The extracted token was `"W1234567"`,
   which never exact-matched the actual argument value `"#W1234567"`.
2. The regex required a 4-character minimum. A test fixture with a
   shorter, but still realistic-length, ID (`"#O2"`) still failed
   after fix #1 alone, which is what surfaced this second edge case.

Both are symptoms of the same design flaw: extracting a fixed set of
"tokens" via regex and checking exact membership is inherently
brittle to whatever the ID format happens to look like. Fixed by
dropping token extraction entirely and checking **direct substring
containment** of the exact argument value against each successful
result's raw content (`is_grounded_in` in `failure_analysis.py`) — no
token boundaries, no character-class special-casing, no length
minimum. `test_legitimate_grounding_same_and_later_turns` (see
`REGRESSION_TESTS.md`) locks this down: it failed twice (once per edge
case) before the substring-based fix made it pass. `audit_full.py` had
the identical `TOKEN_RE` and the identical bug — fixed the same way,
for the same reason, so its tables below reflect the corrected
mechanism too.

**Practical effect, this time much larger**: any conversation where an
order_id got legitimately revealed by a mid-conversation successful
lookup was substantially under-counted. `qwen_task2` moved from
71.0% to 48.4% ungrounded; `llama_task0` moved from 86.7% to 46.7%
ungrounded — the largest single correction in this whole audit
history, well outside the 1-6 point range AUDIT 2 found for rule
choice alone. The matching *mechanism* was the dominant source of
error, not the denominator or the rule variant.

## AUDIT 1 — denominator: does `_id`/`_ids` catch every identifier argument?

Retail domain tool schema, fetched from `tau2-bench` source at the
pinned commit (`src/tau2/domains/retail/tools.py`,
`fc0055dc4e0a316c3f83133267fbd6faaa770992`) — every public tool and
its arguments:

| tool | arguments |
|---|---|
| calculate | expression |
| cancel_pending_order | order_id, reason |
| exchange_delivered_order_items | order_id, item_ids, new_item_ids, payment_method_id |
| find_user_id_by_name_zip | first_name, last_name, **zip** |
| find_user_id_by_email | **email** |
| get_order_details | order_id |
| get_product_details | product_id |
| get_item_details | item_id |
| get_user_details | user_id |
| list_all_product_types | (none) |
| modify_pending_order_address | order_id, address1, address2, city, state, country, **zip** |
| modify_pending_order_items | order_id, item_ids, new_item_ids, payment_method_id |
| modify_pending_order_payment | order_id, payment_method_id |
| modify_user_address | user_id, address1, address2, city, state, country, **zip** |
| return_delivered_order_items | order_id, item_ids, payment_method_id |
| transfer_to_human_agents | summary |

**Gap confirmed**: `email`, `zip`, `first_name`, `last_name` are real
identifying arguments the original `_id`/`_ids`-suffix heuristic
missed entirely. Checked whether the observed transcripts actually
use them (`audit_denominator.py`) — yes: `find_user_id_by_email` and
`find_user_id_by_name_zip` are called in 4 of the 5 conversations
(qwen_task1, qwen_task2, qwen_task3, llama_task0).

**Narrow vs wide denominator, per conversation** (causal, success-only
grounding rule in both cases, substring-matching fix applied — final
mechanism throughout this table):

| conversation | narrow (`_id`/`_ids` only) | wide (+email/zip/first_name/last_name) |
|---|---|---|
| qwen_task0 | 16/16 (100%) | 16/16 (100%) |
| qwen_task1 | 20/20 (100%) | 23/23 (100%) |
| qwen_task2 | 12/28 (42.9%) | 15/31 (48.4%) |
| qwen_task3 | 8/9 (88.9%) | 15/16 (93.8%) |
| llama_task0 | 3/11 (27.3%) | 7/15 (46.7%) |
| **pooled (5 convos)** | 59/84 (70.2%) | 76/101 (75.2%) |

**Widening the denominator moves the pooled rate by about 5 points
(70.2% → 75.2%)** — a bit more than first estimated with the
still-buggy matching mechanism (that earlier estimate said ~2.4
points), but still a secondary effect next to the matching-mechanism
fix itself (see below), which is what actually explains most of the
movement from the original headline.

## AUDIT 2 — robustness of the grounding rule (wide denominator, 3 variants)

| conversation | a) causal + success-only | b) causal + success-and-error | c) non-causal + success-only |
|---|---|---|---|
| qwen_task0 | 16/16 | 15/16 | 16/16 |
| qwen_task1 | 23/23 | 23/23 | 23/23 |
| qwen_task2 | 15/31 | 15/31 | 12/31 |
| qwen_task3 | 15/16 | 15/16 | 15/16 |
| llama_task0 | 7/15 | 7/15 | 4/15 |
| **pooled** | **76/101 (75.2%)** | 75/101 (74.3%) | 70/101 (69.3%) |

- **(a) vs (b), error content included**: barely moves the number
  (75.2% → 74.3%, ~1 point). Excluding error echoes was the right call
  methodologically but wasn't doing much of the headline's work.
- **(a) vs (c), causal vs anywhere-in-conversation**: moves more
  (75.2% → 69.3%, ~6 points) — same direction and similar magnitude as
  before the matching-mechanism fix, confirming this sensitivity is a
  property of the *rule choice*, not an artifact of the bug that got
  fixed alongside it.

**Conclusion on robustness, restated after both bugs**: rule choice
(error-inclusion, causal-vs-anywhere) and denominator choice
(narrow-vs-wide) move the number by single digits, consistently, both
before and after the matching-mechanism fix. The matching mechanism
itself — not any of the above — was the dominant source of error:
qwen_task2 moved 22 points (71.0% → 48.4%) and llama_task0 moved 40
points (86.7% → 46.7%) from that one fix alone, far outside the 1-6
point range everything else moves the number by.

## Corrected Q1/Q3 numbers (wide denominator, rule (a), all three fixes applied) — supersedes both earlier versions of FAILURE_ANALYSIS.md

**Qwen, pooled across 4 tasks**: **69/86 (80.2%)** of ID arguments
ungrounded; **38/43 (88.4%)** of ID-bearing calls had ≥1 ungrounded ID.
Origins: 57 user_utterance (82.6%), 12 invented (17.4%), 0 policy —
essentially unchanged from the pre-matching-fix pass; the fix mostly
changed *how many* IDs are ungrounded, not *where* the remaining
ungrounded ones trace to.

**Llama, task0 only**: **7/15 (46.7%)** ungrounded; **4/10 (40%)** of
calls affected — down sharply from 86.7%/90% in the previous revision,
which was itself already a correction from the first pass's
90.9%/100%. Origins: **3 user_utterance (42.9%), 4 invented (57.1%)**
— still invention-leaning but now on a small, roughly-balanced base
(7 ungrounded IDs, not 10 or 13). **The Qwen-vs-Llama contrast this
document was built around is now much less clear on 1 task**: Llama's
overall ungrounded rate (46.7%) is well below Qwen's pooled rate
(80.2%), which is itself a different, and arguably more interesting,
finding than the original "comparable rate, opposite mechanism" story
— but it rests on n=1 for Llama and needs tasks 1/2/3 before it means
anything.

Q4 (post-error next action: 55.3% lookup tool / 18.4% same-tool-new-args
/ 7.9% exact repeat / 7.9% other / 10.5% no next call, out of 38 error
events) is confirmed **unaffected by any of the three fixes** — that
script never used the grounding pool, it pairs consecutive tool calls
by index regardless of argument shape. Unchanged, still trustworthy.

## Regression tests

All of the above is now covered by `test_failure_analysis.py`
(5/5 passing, see `REGRESSION_TESTS.md` for the full case list and
what each one guards against) — written after the second bug, so it
locks down all three fixes rather than just the last one found.

## What's still open — update: Llama tasks 1/2/3 attempted, 1 of 3 landed

Llama task1 landed with a real transcript; tasks 2 and 3 both failed
for real, unrelated infra reasons (vLLM parser rejected a parallel
tool call on task2; missing `OPENAI_API_KEY` for tau2's NL-assertion
evaluator on task3 — see `FAILURE_ANALYSIS.md`'s data-gap section for
detail). Net: Llama now has 2 of 4 tasks, not the full 4, but the
second data point is informative on its own — Q2's rate gap (Llama
43.9% vs Qwen 80.2%, pooled) and Q3's origin contrast (Llama
76.0% invented vs Qwen 82.6% user-sourced) both held up going from 1
to 2 Llama tasks rather than collapsing toward Qwen's numbers, which
is meaningfully better evidence than the single-task version of this
document had. Still short of Qwen's 4-task sample size.
