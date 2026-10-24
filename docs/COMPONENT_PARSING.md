# A5: component-level recompute of Tables 1/3 for Llama's stringified-list IDs

Written 2026-10-12, reviewer response. `src/component_level_grounding.py`
is a SEPARATE analysis - it does not change `failure_analysis.py`'s default
classifier, imports `is_grounded_in`/`classify_origin`/`load_messages`
unchanged, and writes its own `component_level_grounding_raw.json`.

## Correction to the prior report

`docs/PROVENANCE.md` said "18 of the 35 llama invented id-args" are
stringified lists. **That undercounts.** It missed `llama_task3`'s two
single-element list-strings (`item_ids`/`new_item_ids` = `"[9647292434]"`,
turn 14) - a single-element bracketed string is still a stringified list.
**The real count is 20, not 18.** `docs/PROVENANCE.md` is corrected below;
the number 18 should not be used going forward.

## All 20 stringified-list invented id-args, llama tasks 0/1/3

| conversation | turn | field | value |
|---|---|---|---|
| llama_task0 | 4 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 4 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task0 | 12 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 12 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task0 | 16 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 16 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task0 | 22 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 22 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task0 | 28 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 28 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task0 | 50 | item_ids | `[1151293680, 4983901480]` |
| llama_task0 | 50 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task1 | 16 | item_ids | `[1151293680, 4983901480]` |
| llama_task1 | 16 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task1 | 26 | item_ids | `[1151293680, 4983901480]` |
| llama_task1 | 26 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task1 | 40 | item_ids | `[1151293680, 4983901480]` |
| llama_task1 | 40 | new_item_ids | `[1151293681, 4983901481]` |
| llama_task3 | 14 | item_ids | `[9647292434]` |
| llama_task3 | 14 | new_item_ids | `[9647292434]` |

3 distinct values: `[1151293680, 4983901480]` -> components
`1151293680`, `4983901480`; `[1151293681, 4983901481]` -> `1151293681`,
`4983901481`; `[9647292434]` -> `9647292434`.

**Qwen confirmed to have zero stringified-list id-args**, checked across
all 4 qwen tasks (not just 0/1/3), by `component_level_grounding.py`'s
`find_all_stringified_list_values`.

## What each component actually is, once split

- `item_ids` components (`1151293680`, `4983901480`, `9647292434`): the
  ALREADY-OWNED item/order values. Grounded via an earlier successful
  `get_order_details`/`get_item_details` result in every single
  occurrence (19/19 component-instances). The merged bracket-string never
  matched that result; the individual number always did.
- `new_item_ids` components (`1151293681`, `4983901481`): the PROPOSED
  replacement items, never returned by any tool. Never grounded (correctly
  - nothing should ground them, they don't exist yet). Origin: `invented`
  at `llama_task0` turns 4/12 and `llama_task1` turn 16 (the user hadn't
  typed them yet at that causal point); `user_utterance` at every later
  occurrence (turns 16/22/28/50 in task0, 26/40 in task1) - the user types
  both numbers in plain text, repeatedly, confirmed in `docs/PROVENANCE.md`'s
  hand audit.
- `llama_task3`'s single-element lists: both components (`9647292434`)
  grounded via the `get_item_details` success at turn 10.

## Old (default classifier) vs new (component-parsed) - llama tasks 0/1/3

| | OLD (failure_analysis.py) | NEW (component_level_grounding.py) |
|---|---|---|
| FLAT ungrounded | 47/79 (59.5%) | 45/97 (46.4%) |
| FLAT origins | user 12 (25.5%), invented 35 (74.5%) | user 24 (53.3%), invented 21 (46.7%) |
| CONDITIONAL ungrounded | 45/77 (58.4%) | 43/95 (45.3%) |
| CONDITIONAL origins | user 10 (22.2%), invented 35 (77.8%) | user 22 (51.2%), invented 21 (48.8%) |

Per task (FLAT ungrounded, old -> new):

| task | old | new |
|---|---|---|
| task0 | 23/39 | 23/51 |
| task1 | 16/30 | 16/36 |
| task3 | 8/10 | 6/10 |

The ungrounded COUNT barely moves for task0/task1 (23->23, 16->16) because
`item_ids` components all flip to grounded while `new_item_ids` components
stay ungrounded but mostly flip from `invented` to `user_utterance` - a
wash on the rate, a real move on provenance. task3's count drops (8->6)
because both of its single-element components turn out to be grounded.

For reference, Qwen's conditional-ungrounded provenance (tasks 0/1/3,
unaffected by this analysis - it has no stringified lists): 14/14 ungrounded
ids are `user_utterance` (100%), 0 `invented`.

## Plain answer

**Does Llama still mostly invent?** No, not under component-level parsing.
The default (whole-argument) method says 74.5-77.8% invented - a clear
majority. Once the schema-malformed lists are decomposed, Llama's
ungrounded provenance (tasks 0/1/3) is close to an even split - 46.7-48.8%
invented, 51.2-53.3% user_utterance - a slight USER-STATED majority, not an
invented majority. Over half of what the default method counted as
"invented" turns out, at the component level, to be either (a) correctly
grounded via an earlier tool lookup, or (b) verbatim, repeated user
statements wrapped in a serialization format that happens not to match
literally.

**Does the Qwen-vs-Llama provenance contrast survive?** Yes, in direction,
not in the "mostly invents" characterization. Qwen is still 100%
user_utterance / 0% invented on its conditional-ungrounded ids - genuinely
never invents. Llama still invents far more than Qwen (46.7-48.8% vs 0%) -
the DIRECTION of the contrast holds. But "Llama mostly invents" should not
be the paper's wording for the component-parsed view; "Llama invents about
as often as Qwen never does, roughly half the time" is the accurate
description under this alternate analysis. The default (whole-argument)
number remains 74.5-77.8% invented and is still the number this repository
actually computes by default - this document reports the alternate
component-level view alongside it, per the reviewer's request, without
changing which one `failure_analysis.py` produces.

## Not done

No number in `failure_analysis_raw.json`, `matched_comparison_raw.json`, or
any paper-cited table was changed. `src/component_level_grounding.py` is
new, separate, and writes its own output file.
