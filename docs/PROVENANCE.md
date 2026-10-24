# Provenance classification: exact rule, normalization, and hand audit

Written 2026-10-12, reviewer response. Covers `classify_origin` and
`is_grounded_in` in `src/failure_analysis.py` exactly as implemented - this is
a description of the real code, not a spec for how it should work. Table 3's
numbers are reproduced from `failure_analysis_raw.json` with no changes to
the analysis code.

## Grounding rule (is_grounded_in)

```python
def is_grounded_in(value: str, contents: list[str]) -> bool:
    return any(value in c for c in contents)
```

A value is **grounded** if it is an exact substring of the text of some
EARLIER **successful** tool result in the same conversation (causal: only
results already returned before this call count - see
`test_causal_ordering_future_result_does_not_ground_past_use`). Error results
never ground anything, even if they echo the value back
(`test_id_only_in_error_result_stays_ungrounded`).

## Provenance rule for ungrounded values (classify_origin)

```python
def classify_origin(value, policy_text, user_texts) -> str:
    if value in policy_text:
        return "policy_document"
    for t in user_texts:
        if value in t:
            return "user_utterance"
    return "invented"
```

**Actual check order: policy -> user -> invented.**

This is the opposite of what's commonly assumed when describing this script
informally ("user -> policy -> invented") - flagging that explicitly here
since the two orders can disagree whenever a value appears in both the
policy text and a user utterance. `test_classify_origin_policy_beats_user_utterance`
locks in the real order: policy wins ties.

`user_texts` is the list of user message strings seen so far in the forward
pass over `messages` - causal, same as the grounding pool. `policy_text` is
the whole policy document, loaded once, not causally restricted (a value
"in the policy" at turn 2 is still "in the policy" at turn 50).

## Normalization: none

The match is Python's `in` operator on the raw strings. No normalization is
applied:

- **Case**: `"abc"` and `"ABC"` do not match each other.
- **Whitespace**: internal or surrounding whitespace differences are not
  collapsed.
- **Quotes**: a value wrapped in backticks or quotes in the user's markdown
  (common in these transcripts - users frequently write `` `1008292230` ``)
  does not need special handling in the CURRENT direction only because
  `value in text` checks whether `value` sits inside `text` as a substring -
  the backticks around `` `1008292230` `` don't break that. The quote
  sensitivity bites the other direction: if the VALUE itself carried quote
  characters that the user's text didn't, it would fail. No case in the
  audited data triggers this; see `test_classify_origin_exact_user_match` for
  the backtick case working correctly today, by substring luck rather than
  explicit normalization.
- **`#` prefix**: not stripped either direction. `"#W123" in "...W123..."` is
  `False` unless the literal `#` is also present in the text. This is a real,
  demonstrated gap - see `test_classify_origin_hash_prefix_not_normalized`.
  It did **not** change any Table 3 number in the audited data (see below);
  every `#`-prefixed ID in the tasks 0/1/3 transcripts happens to be stated by
  the user with the same `#` already attached, so the gap is latent, not
  currently load-bearing.

## Hand audit: every id classified "invented", both models, tasks 0/1/3

46 total invented id-args audited (qwen: 7+2+2=11 across tasks 0/1/3; llama:
19+9+7=35). For each, checked:

1. Exact substring in any user turn, across the WHOLE conversation (not just
   causally - deliberately looser than the production check, to catch
   anything the causal check could have missed).
2. Same check after normalizing case/whitespace/quotes.
3. Same check again after additionally stripping a leading `#`.
4. Substring in the policy text.

**Agreement: 46/46.** Zero cases where the hand audit (even with the three
normalizations above) finds the value actually stated by the user or present
in policy while the code says "invented". **Disagreements: none.**

Three values showed up as a hit on the full-conversation (non-causal) scan
but are NOT disagreements - they are the expected "agent says it first, user
says the same string back later" pattern, already working as designed
because the production check is causal:

| conversation | value | invented at | becomes user_utterance at |
|---|---|---|---|
| qwen_task0 | `LOGITHEK-GPROX` (item_id) | turn 23 | turn 28 |
| llama_task0 | `yusuf_rossi_9620@example.com` (email) | turn 36 | turn 40 |
| llama_task3 | `#W0000000` (order_id) | turn 14 | turn 18 |

In each case the agent used the exact value before the user had typed it
(correctly "invented" at that point), and the same value is correctly
reclassified "user_utterance" once the user does type it later in the same
conversation. This is the causal check working as intended, not a bug.

### Caveat found during the audit: schema-malformed list arguments hide real grounding

`schema_conformance.py` already shows Llama submits `item_ids`/`new_item_ids`
as a single malformed STRING that merely *looks* like a list (e.g.
`"[1151293680, 4983901480]"`) instead of a real JSON list, in 10/10 of its
malformed calls (tasks 0/1/3). `id_values()` only decomposes a field into
multiple candidate IDs when the JSON value is an actual `list`; a string
that looks like one is kept as a single opaque value.

Effect on provenance: that single garbled string (brackets, comma, the
literal word formatting the agent chose) essentially never matches a policy
or user substring, even when the individual numbers inside it plainly do.
This accounts for 12 of llama_task0's 19 invented id-args (`item_ids` x6 +
`new_item_ids` x6), 6 of llama_task1's 9 (`item_ids` x3 + `new_item_ids`
x3), and 2 of llama_task3's 7 (`item_ids` x1 + `new_item_ids` x1, the
single-element `"[9647292434]"` case) - **20** of the 35 llama invented
id-args, over half. (Corrected 2026-10-12: an earlier draft of this
document said 18, missing llama_task3's single-element case - see
`docs/COMPONENT_PARSING.md` for the full recount and the component-level
Table 1/3 recompute.)

- `llama_task0`'s `item_ids`/`new_item_ids` values reference IDs
  (`1151293680`, `4983901480`, `1151293681`, `4983901481`) that the user DID
  state as plain numbers (u[2], u[3]) - just never in the bracketed,
  comma-joined form the agent submitted.
- `llama_task1`'s `new_item_ids="[1151293681, 4983901481]"` is the clearest
  case: both numbers appear verbatim, repeatedly, as plain text in the
  user's own messages (u[2] through u[6]) - not reconstructed, not inferred,
  typed directly by the user simulator. The merged bracket-string the agent
  submitted still doesn't match any single utterance as a substring, so it's
  "invented" under the current method.

This is **not** scored as a disagreement above: per the method as actually
implemented, the exact submitted value genuinely doesn't appear anywhere
(checked with normalization too), so "invented" is the correct mechanical
answer. But it means part of the "invented" bucket - specifically, for a
model that also has 100% schema-malformation on list arguments - is "real,
correctly-sourced information serialized into a form that fails a literal
match," not "fabricated from nothing." Table 3's invented/user split should
be read with this caveat; a future version of this method that decomposed
malformed list-strings before classification would likely move some of
these 18 id-args from "invented" to "user_utterance", which would shrink
(not reverse) the invented-majority gap for Llama. No number in this
repository has been changed based on this caveat - flagging it for the
paper to describe accurately, not fixing the method mid-review.

## Regression tests added

`tests/test_failure_analysis.py`: `test_classify_origin_exact_user_match`,
`test_classify_origin_policy_beats_user_utterance`,
`test_classify_origin_hash_prefix_not_normalized`,
`test_classify_origin_true_invention`. All pass (11 -> 15/15 at the
time these 4 were added; the suite has since grown to 16/16 with a
later component-level test, see `docs/REGRESSION_TESTS.md` for the
current full list).
