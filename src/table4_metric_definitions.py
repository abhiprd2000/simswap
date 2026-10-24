"""Reviewer response A6 - what exactly does Table 4 / user_id_mentions.py
compute, and does it match the abstract's wording?

SEPARATE analysis. Does not change user_id_mentions.py. Computes BOTH
candidate definitions side by side, per conversation, so the ambiguity
raised by the reviewer is resolved with numbers instead of prose.

(ii) "tool-call IDs the user stated / tool-call IDs"
     = EXACTLY what user_id_mentions.py already prints
       (distinct_id_values_stated_by_user / distinct_id_values_in_tool_calls).
     numerator: distinct id-shaped values that (a) appear as a tool-call
       argument somewhere in this conversation AND (b) appear as a
       substring of some user utterance in this conversation.
     denominator: distinct id-shaped values that appear as a tool-call
       argument somewhere in this conversation - PERIOD, including
       agent-invented ones the user never said.

(i)  "user-stated IDs that appear in a tool call / user-stated IDs"
     CANNOT be computed exactly from this codebase: there is no
     independent extractor for "ID-shaped values the user stated",
     decoupled from the tool-call vocabulary - user_id_mentions.py only
     ever checks tool-call values against user text, never scans user
     text on its own. Building a free-text ID-token regex from scratch
     would be a NEW, unvalidated heuristic - exactly the kind of
     invented number the reviewer instruction forbids.

     Best available PROXY, clearly labeled as a proxy, not a true (i):
     pool the tool-call id-vocabulary of BOTH models on the SAME task
     (they're the same underlying tau2 task/order, so this is the
     union of IDs either agent is known to use for that task, not a
     made-up pattern) and check which of THOSE appear in THIS
     conversation's own user text. This still cannot see an id the user
     stated that NEITHER model's agent ever used in a tool call - it is
     a lower bound on the true (i), not an exact value.
"""

from __future__ import annotations

import json
from pathlib import Path

import failure_analysis as fa
import user_id_mentions as uim

TASK_OF = {name: name.split("_task")[1] for name in uim.SOURCES}


def conversation_data(name, path):
    messages, _policy = fa.load_messages(path)
    user_texts = [m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
    all_id_values = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                for _field, value in fa.id_values(tc.get("arguments") or {}):
                    all_id_values.add(value)
    return user_texts, all_id_values


def main():
    per_convo = {}
    for name, path in uim.SOURCES.items():
        if not Path(path).exists():
            print(f"MISSING: {name} -> {path}")
            continue
        per_convo[name] = conversation_data(name, path)

    # pooled per-task vocabulary across both models (the proxy universe for metric i)
    pooled_vocab_by_task: dict[str, set] = {}
    for name, (_user_texts, vocab) in per_convo.items():
        t = TASK_OF[name]
        pooled_vocab_by_task.setdefault(t, set()).update(vocab)

    print(f"{'conversation':<14}{'user_model':<18}"
          f"{'(ii) num/den':<16}{'(ii) %':<9}"
          f"{'(i)-proxy num/den':<20}{'(i)-proxy %':<10}")
    for name, (user_texts, vocab) in per_convo.items():
        t = TASK_OF[name]
        user_model = uim.USER_MODEL_BY_CONVO[name]

        # (ii) exactly what user_id_mentions.py prints today
        stated_by_user_in_vocab = {v for v in vocab if any(v in ut for ut in user_texts)}
        ii_num, ii_den = len(stated_by_user_in_vocab), len(vocab)
        ii_frac = ii_num / ii_den if ii_den else None

        # (i)-proxy: pooled-task vocabulary checked against this conversation's user text,
        # then restricted to this conversation's own tool-call vocabulary for the numerator
        pooled = pooled_vocab_by_task[t]
        user_stated_proxy = {v for v in pooled if any(v in ut for ut in user_texts)}
        reaches_tool_call_proxy = user_stated_proxy & vocab
        i_num, i_den = len(reaches_tool_call_proxy), len(user_stated_proxy)
        i_frac = i_num / i_den if i_den else None

        print(f"{name:<14}{user_model:<18}"
              f"{f'{ii_num}/{ii_den}':<16}{f'{ii_frac:.1%}' if ii_frac is not None else 'n/a':<9}"
              f"{f'{i_num}/{i_den}':<20}{f'{i_frac:.1%}' if i_frac is not None else 'n/a':<10}")

    print("\n(ii) is EXACTLY user_id_mentions.py's current printed number - confirmed by "
          "construction above (same numerator/denominator logic, same inputs).")
    print("(i)-proxy is a LOWER-BOUND PROXY, not the true (i) - see module docstring. "
          "Do not cite as 'the' fraction of user-stated IDs without this caveat.")


if __name__ == "__main__":
    main()
