# Schema conformance — a second, orthogonal failure axis

**Note, 2026-08-17:** Llama's transcripts below are the ORIGINAL data
(fixed Qwen2.5-1.5B user, see `FAILURE_ANALYSIS.md`'s "CONFOUND CHECK"
section). Rechecked against the matched-user-simulator rerun
(Qwen2.5-7B-AWQ user for both models): result unchanged qualitatively,
Qwen 0/8, Llama now 10/10 malformed calls (was 9/9) — still binary, the
extra list-bearing call comes from the new, longer/different task0
conversation. See `matched_usersim_pooled.json`'s `schema_conformance`
block for the exact matched-user-sim numbers; not rewritten in place
here since the qualitative finding (0% vs 100%) is identical either way.

Written 2026-08-16, on operator instruction, from the conversations
already on disk (now 7: 4 Qwen, 3 Llama) — no new GPU runs. This is
independent of grounding (`FAILURE_ANALYSIS.md`): grounding asks
"was this ID ever confirmed by a tool," this asks "did the agent even
send the argument in the shape the tool expects." A call can fail
either way, or both, or neither.

## Method

Real retail tool schema (`AUDIT_REPORT.md` AUDIT 1, fetched from
tau2-bench source): the **only** `List[str]`-typed arguments in the
whole domain are `item_ids` and `new_item_ids` (used by
`exchange_delivered_order_items`, `modify_pending_order_items`,
`return_delivered_order_items`). Every other argument in every tool is
a plain `str`. `schema_conformance.py` checks every tool call's
argument JSON types against this schema — is `item_ids`/`new_item_ids`
an actual list, or something else (a string, a number)? Is any
non-list field wrongly a list? This is a structural type check, not a
guess from error-message text.

This is now TWO checks at two different levels, both under the
"schema conformance" umbrella (2026-08-16, second round): (1)
argument-type malformation, below - a real tool call went through,
but an argument has the wrong JSON type; (2) request-structure
rejection (`NON_TRANSCRIPT_EVENTS` in `schema_conformance.py`) - the
whole request never became a transcript entry, because vLLM's parser
rejected it before tau2 ever saw a result. Reported as separate counts
with different denominators (a per-argument rate vs. a per-attempt
yes/no) - merging them would misstate both. Covered by 2 of the 9
regression tests (`test_schema_conformance_detects_str_instead_of_list`,
`test_schema_conformance_records_llama_parallel_tool_call_rejection`,
`REGRESSION_TESTS.md`).

## Result — axis 1: argument-type malformation

| model | malformed calls / calls to list-bearing tools | rate |
|---|---|---|
| Qwen (4 tasks) | 0 / 8 | **0.0%** |
| Llama (3 tasks) | 9 / 9 | **100.0%** |

Every single time Llama attempted a tool call with `item_ids` or
`new_item_ids` — across all 3 sampled tasks, 9 opportunities total —
it sent a **string that merely looks like a list** (e.g.
`"[1151293680]"` — a JSON string containing bracket characters, not an
actual array) instead of a real list. Every single time Qwen attempted
the same (8 opportunities across 4 tasks), it sent a correctly-typed
list. This is completely binary and now holds across a third
independent Llama task — not "Llama does it more often," but "Llama
always does it, Qwen never does."

Concrete example (`llama_task1`):
```
CALL exchange_delivered_order_items(
    order_id='#W2378156',
    item_ids='[1151293680]',        <- string, not a list
    new_item_ids='[1656367028]',    <- string, not a list
    payment_method_id='credit_card_9513926')
RESULT error=True content='Error: Number of [ not found.'
```
The error message itself is a symptom of the type mismatch — tau2's
tool implementation appears to iterate the string character-by-character
(or otherwise mis-parse it), producing `"["` as if it were an item ID.
This happened in `llama_task0` (1 occurrence), `llama_task1` (7
occurrences — the agent never corrects to a real list even after
repeated errors), and `llama_task3` (1 occurrence, on
`modify_pending_order_items` this time, not just `exchange_delivered_order_items`
— confirming the malformation isn't specific to one tool).

## Result — axis 2: request-structure rejection (no transcript to read)

| model | rejections | detail |
|---|---|---|
| Qwen | 0 | — |
| Llama | 2 independent attempts, both `llama_task2` | vLLM's `llama3_json` parser 400s the whole request: `"This model only supports single tool-calls at once!"` |

This isn't a malformed *argument* — the request itself is structurally
invalid from the parser's point of view: Llama attempted to emit
**multiple simultaneous `tool_calls` in one turn**, and the
`llama3_json` parser has no graceful degradation path for that (unlike
`hermes`, which tolerated whatever Qwen did across all 4 of its
tasks). Nothing is written to the transcript when this happens — tau2
never gets a response back to record — so this can't be counted by
walking `sim_dump.messages` like axis 1 above. It's recorded directly
from the run's own error traceback
(`kaggle_output_llama_task23/result_llama_task2.json`,
`NON_TRANSCRIPT_EVENTS` in `schema_conformance.py`) and locked down by
`test_schema_conformance_records_llama_parallel_tool_call_rejection`
so it can't be silently lost in a future edit — there is no
transcript-based check that would ever rediscover it. Confirmed
identical on 2 independent GPU attempts (see `PROGRESS.md`), so this
is real, reproducible model/parser behavior, not a fluke.

## Why this matters as a second axis, not a grounding artifact

This is a distinct failure mode from ungrounded IDs: even a
**perfectly grounded** item_id (one that came straight from a
successful lookup) still fails if it's packaged in the wrong container
type. None of this file's axis-1 (9 malformed calls) or axis-2 (2
rejected requests) events are counted or double-counted in
`FAILURE_ANALYSIS.md`'s grounding numbers — three independent passes
over the same transcripts, three different questions. A model could
plausibly be good at one axis and bad at another; in this sample,
Llama is worse than Qwen on all three (lower absolute grounding *and*
schema-malformed calls *and* request-structure rejections where Qwen
has none of either), but they're not the same underlying problem and
shouldn't be conflated into a single number.

## What this does and doesn't establish

17 malformed-eligible calls total (9 for Llama across 3 tasks, 8 for
Qwen across 4) — a completely clean 0/8 vs 9/9 split, now confirmed on
a third independent Llama task (`modify_pending_order_items`, not just
`exchange_delivered_order_items` — same malformation, different tool),
is stronger evidence than the original 2-task version but still a
small sample overall. Task2's failure (vLLM parser rejecting a
parallel tool call) means this can't be checked on a 4th Llama task
without a different task or a different approach to task2's parser
issue. Worth checking whether this is specific to `item_ids`/
`new_item_ids` serialization under this exact vLLM parser/chat-template
combination (`llama3_json` + `tool_chat_template_llama3.1_json.jinja`)
or a broader Llama-3.1-8B-AWQ tool-calling quirk that might not
generalize to other Llama deployments.
