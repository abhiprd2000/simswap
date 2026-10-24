# A6: what Table 4 / user_id_mentions.py actually measures

Written 2026-10-12, reviewer response. `src/table4_metric_definitions.py`
is a separate analysis; it does not change `user_id_mentions.py`.

## The abstract's wording vs. what the code computes

The abstract says: "the fraction of a simulator's stated identifiers that
reach a tool call." Call this **(i)**: `user-stated IDs that appear in a
tool call / user-stated IDs`.

`user_id_mentions.py` actually computes **(ii)**: `tool-call IDs the user
stated / tool-call IDs` - the reverse direction.

```python
all_id_values = {every distinct id-shaped value that appears as a
                  tool-call argument anywhere in this conversation}
stated_by_user = {v for v in all_id_values if any(v in t for t in user_texts)}
# printed: len(stated_by_user) / len(all_id_values)
```

**Numerator** (`distinct_id_values_stated_by_user`): distinct values that
are BOTH (a) a tool-call argument somewhere in the conversation AND (b) a
literal substring of some user utterance.

**Denominator** (`distinct_id_values_in_tool_calls`): every distinct value
that is a tool-call argument anywhere in the conversation - full stop,
including values the agent invented and the user never said.

This is exactly (ii), not (i). The per-conversation "distinct IDs stated"
counts cited as 5, 8, 7, 4 (7B) and 3, 3, 0 (1.5B) are **(ii)'s
numerators** - they count only tool-call-vocabulary members that the user
also happened to state, NOT every ID-shaped value the user said in the
conversation. There is no field anywhere in this codebase for "every ID
the user stated, independent of whether it was ever used in a tool call" -
`user_id_mentions.py` never scans user text on its own; it only checks
tool-call values against user text.

## Why (i) can't be computed exactly from this codebase

(i)'s denominator needs an independent extractor: "every ID-shaped value
the user stated," decoupled from what the agent later did with it. No such
extractor exists anywhere in `src/`. The field-name heuristic used
elsewhere (`NARROW_FIELD_RE`/`WIDE_FIELD_NAMES` in `failure_analysis.py`)
identifies ID-shaped ARGUMENT FIELDS by name - it has nothing to say about
free-text user messages, which carry no field names. Building a
from-scratch regex to spot "ID-shaped" substrings in prose (digit runs?
order-id patterns? capitalized name pairs?) would be a brand-new,
unvalidated heuristic whose result depends entirely on how it's written -
exactly the kind of invented number this task instruction forbids.
Not computed as a made-up true value for that reason.

## Both metrics, computed per conversation

(ii) is exact (identical construction to the production code). (i) is
replaced with a clearly-labeled PROXY: pool the tool-call id-vocabulary of
BOTH models on the same underlying task (same tau2 order/items either
agent might reference - not an invented pattern, just the union of two
already-computed vocabularies), then check which of those pooled values
this conversation's user text states, and how many of those also appear in
this conversation's own tool calls. This is a lower bound on the true (i)
- it cannot see an id the user stated that NEITHER model's agent ever used
in any tool call.

| conversation | user model | (ii) num/den | (ii) % | (i)-proxy num/den | (i)-proxy % |
|---|---|---|---|---|---|
| qwen_task0 | Qwen2.5-7B-AWQ | 5/11 | 45.5% | 5/5 | 100.0% |
| qwen_task1 | Qwen2.5-7B-AWQ | 8/9 | 88.9% | 8/9 | 88.9% |
| qwen_task2 | Qwen2.5-7B-AWQ | 7/13 | 53.8% | 7/7 | 100.0% |
| qwen_task3 | Qwen2.5-7B-AWQ | 4/7 | 57.1% | 4/5 | 80.0% |
| llama_task0 (1.5B) | Qwen2.5-1.5B-fp16 | 3/14 | 21.4% | 3/3 | 100.0% |
| llama_task1 (1.5B) | Qwen2.5-1.5B-fp16 | 3/16 | 18.8% | 3/3 | 100.0% |
| llama_task3 (1.5B) | Qwen2.5-1.5B-fp16 | 0/9 | 0.0% | 0/0 | n/a |

The (ii) column reproduces the exact numbers already reported (5/11, 8/9,
7/13, 4/7, 3/14, 3/16, 0/9) - confirmed by construction, not just by
re-reading the old output.

The (i)-proxy column is near-ceiling for most rows (the pooled-vocabulary
restriction already biases the denominator toward values that tend to get
used in a tool call somewhere) and is **not a usable estimate of the true
(i)** - it is reported only because the task asked for both, with this
caveat attached. Do not put the (i)-proxy numbers in the paper as "the"
fraction without this caveat; they are a lower bound, not a measurement.

## Which one is "the current number"

The 3.0x figure (Qwen mean 6.00 vs 1.5B mean 2.00) and the 0-21%/46-89%
ranges already reported are all built from **(ii)**: tool-call IDs the
user also stated, as a fraction of all tool-call IDs. If the abstract's
sentence ("fraction of a simulator's stated identifiers that reach a tool
call") is meant literally, it describes (i), not what's actually measured.
Two fixes, pick one for the paper - not performed here, since it's a
wording decision, not a data question:
- Reword the abstract/Table 4 caption to match (ii): "fraction of a
  simulator's TOOL-CALL identifier vocabulary that the simulator itself
  had stated" (i.e., describe the denominator as tool-call IDs, not
  user-stated IDs).
- Or compute a defensible (i) with a newly-specified, pre-registered
  extraction rule - not attempted here per "never invent a number."
