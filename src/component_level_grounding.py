"""Reviewer response A5 - component-level recompute of Tables 1/3 for
Llama's stringified-list id arguments.

SEPARATE analysis. Does not change failure_analysis.py's default
classifier or its output files - imports and reuses its primitives
(is_grounded_in, classify_origin, load_messages) unchanged. The only new
piece is parse_stringified_list(), which decomposes a malformed
stringified-list VALUE (e.g. "[1151293680, 4983901480]" - the real llama
schema violation documented in docs/SCHEMA_CONFORMANCE.md) into its
component ID strings, the same way failure_analysis.id_values() already
decomposes a genuine JSON list into separate candidates. Everything
downstream (grounding, provenance, counting) runs the production rules
unchanged on each component. See docs/COMPONENT_PARSING.md for the
numbers and the plain answer to "does Llama still mostly invent".
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import failure_analysis as fa

STRINGIFIED_LIST_RE = re.compile(r"^\[(.*)\]$")


def parse_stringified_list(value: str) -> list[str] | None:
    """If value is a string shaped like "[a, b, c]", return the component
    strings (whitespace- and quote-stripped). Otherwise None - including
    for the empty-list string "[]", since that has no components to
    reclassify and callers should fall back to treating it as the single
    original value."""
    m = STRINGIFIED_LIST_RE.match(value.strip())
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None
    parts = [p.strip().strip("\"'") for p in inner.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def id_values_component_parsed(args: dict) -> list[tuple[str, str]]:
    """Same contract as failure_analysis.id_values(), but any string value
    that LOOKS like a bracketed list is further split into its component
    ID strings. A field whose value was already a real JSON list is
    untouched (id_values() already decomposed it)."""
    out = []
    for field, value in fa.id_values(args):
        components = parse_stringified_list(value)
        if components:
            for c in components:
                out.append((field, c))
        else:
            out.append((field, value))
    return out


def analyze_conversation_component_parsed(name: str, path) -> dict:
    """Line-for-line the same control flow as
    failure_analysis.analyze_conversation, with id_values() replaced by
    id_values_component_parsed(). Kept as a full copy rather than a
    parameterized call into failure_analysis.py so the default classifier
    file is untouched (reviewer instruction)."""
    messages, policy_text = fa.load_messages(path)

    grounded_contents: list[str] = []
    user_texts: list[str] = []
    tool_calls_log = []
    call_result_queue: list[int] = []

    for i, m in enumerate(messages):
        role = m.get("role")

        if role == "user":
            c = m.get("content")
            if isinstance(c, str):
                user_texts.append(c)

        elif role == "assistant" and m.get("tool_calls"):
            pool_nonempty = len(grounded_contents) > 0
            for tc in m["tool_calls"]:
                args = tc.get("arguments") or {}
                ids = id_values_component_parsed(args)
                id_records = []
                for field, value in ids:
                    is_grounded = fa.is_grounded_in(value, grounded_contents)
                    origin = None if is_grounded else fa.classify_origin(value, policy_text, user_texts)
                    id_records.append({
                        "field": field, "value": value, "grounded": is_grounded,
                        "origin": origin, "pool_nonempty": pool_nonempty,
                    })
                call_index = len(tool_calls_log)
                tool_calls_log.append({
                    "turn": i, "name": tc.get("name"), "ids": id_records,
                    "pool_nonempty": pool_nonempty, "errored": None,
                })
                call_result_queue.append(call_index)

        elif role == "tool":
            content = m.get("content")
            is_error = bool(m.get("error"))
            content_str = content if isinstance(content, str) else json.dumps(content, default=str)
            if call_result_queue:
                call_idx = call_result_queue.pop(0)
                tool_calls_log[call_idx]["errored"] = is_error
            if not is_error:
                grounded_contents.append(content_str)

    total_ids = sum(len(c["ids"]) for c in tool_calls_log)
    ungrounded_ids = sum(1 for c in tool_calls_log for x in c["ids"] if not x["grounded"])
    calls_with_id = sum(1 for c in tool_calls_log if c["ids"])
    calls_with_ungrounded = sum(1 for c in tool_calls_log if any(not x["grounded"] for x in c["ids"]))

    origin_counts = {"policy_document": 0, "user_utterance": 0, "invented": 0}
    for c in tool_calls_log:
        for x in c["ids"]:
            if not x["grounded"] and x["origin"]:
                origin_counts[x["origin"]] += 1

    ids_pool_nonempty = [x for c in tool_calls_log for x in c["ids"] if x["pool_nonempty"]]
    ungrounded_pool_nonempty = sum(1 for x in ids_pool_nonempty if not x["grounded"])
    calls_pool_nonempty = [c for c in tool_calls_log if c["pool_nonempty"] and c["ids"]]
    calls_ungrounded_pool_nonempty = sum(1 for c in calls_pool_nonempty if any(not x["grounded"] for x in c["ids"]))
    origin_counts_conditional = {"policy_document": 0, "user_utterance": 0, "invented": 0}
    for x in ids_pool_nonempty:
        if not x["grounded"] and x["origin"]:
            origin_counts_conditional[x["origin"]] += 1

    return {
        "name": name,
        "total_id_arguments": total_ids,
        "ungrounded_id_arguments": ungrounded_ids,
        "ungrounded_fraction_by_argument": (ungrounded_ids / total_ids) if total_ids else None,
        "calls_with_any_id": calls_with_id,
        "calls_with_ungrounded_id": calls_with_ungrounded,
        "ungrounded_fraction_by_call": (calls_with_ungrounded / calls_with_id) if calls_with_id else None,
        "origin_counts_ungrounded": origin_counts,
        "pool_ever_nonempty": len(grounded_contents) > 0,
        "total_id_arguments_conditional": len(ids_pool_nonempty),
        "ungrounded_id_arguments_conditional": ungrounded_pool_nonempty,
        "ungrounded_fraction_by_argument_conditional": (
            ungrounded_pool_nonempty / len(ids_pool_nonempty) if ids_pool_nonempty else None
        ),
        "calls_with_any_id_conditional": len(calls_pool_nonempty),
        "calls_with_ungrounded_id_conditional": calls_ungrounded_pool_nonempty,
        "ungrounded_fraction_by_call_conditional": (
            calls_ungrounded_pool_nonempty / len(calls_pool_nonempty) if calls_pool_nonempty else None
        ),
        "origin_counts_ungrounded_conditional": origin_counts_conditional,
        "tool_calls_log": tool_calls_log,
    }


def find_all_stringified_list_values(name: str, path) -> list[dict]:
    """Every id-arg value (pre-component-parsing) that matches the
    stringified-list shape, with enough detail to list it: conversation,
    field, raw value, turn, and whether the ORIGINAL (unparsed)
    classification was invented/user/policy/grounded."""
    r = fa.analyze_conversation(name, path)
    out = []
    for c in r["tool_calls_log"]:
        for x in c["ids"]:
            if parse_stringified_list(x["value"]):
                out.append({
                    "conversation": name, "turn": c["turn"], "field": x["field"],
                    "value": x["value"], "original_grounded": x["grounded"],
                    "original_origin": x["origin"],
                })
    return out


MATCHED_TASKS = ("task0", "task1", "task3")


def main() -> None:
    llama_sources = {k: v for k, v in fa.SOURCES.items()
                      if k.startswith("llama_") and k.split("_")[1] in MATCHED_TASKS}
    qwen_sources = {k: v for k, v in fa.SOURCES.items() if k.startswith("qwen_")}

    print("=== confirm Qwen has zero stringified-list id-args (all tasks, not just 0/1/3) ===")
    qwen_hits = []
    for name, path in qwen_sources.items():
        qwen_hits.extend(find_all_stringified_list_values(name, path))
    if qwen_hits:
        print(f"UNEXPECTED: {len(qwen_hits)} stringified-list values found in qwen: {qwen_hits}")
    else:
        print("confirmed: 0 stringified-list id-args in any qwen conversation")

    print("\n=== all 18 stringified-list invented id-args, llama tasks 0/1/3 ===")
    all_listish = []
    for name, path in llama_sources.items():
        all_listish.extend(find_all_stringified_list_values(name, path))
    invented_listish = [x for x in all_listish if x["original_origin"] == "invented" and not x["original_grounded"]]
    for x in invented_listish:
        print(f"  {x['conversation']} turn={x['turn']} field={x['field']!r} value={x['value']!r}")
    print(f"total: {len(invented_listish)} (all stringified-list values found, any origin: {len(all_listish)})")

    print("\n=== component parse of each distinct value ===")
    for val in sorted({x["value"] for x in all_listish}):
        print(f"  {val!r} -> {parse_stringified_list(val)}")

    print("\n=== OLD (default classifier, failure_analysis.analyze_conversation) vs "
          "NEW (component-parsed) - llama tasks 0/1/3 ===")
    old_results = {name: fa.analyze_conversation(name, path) for name, path in llama_sources.items()}
    new_results = {name: analyze_conversation_component_parsed(name, path) for name, path in llama_sources.items()}

    def totals(results, conditional=False):
        suffix = "_conditional" if conditional else ""
        tot = sum(r[f"total_id_arguments{suffix}"] for r in results.values())
        ung = sum(r[f"ungrounded_id_arguments{suffix}"] for r in results.values())
        # scan tool_calls_log directly for both OLD and NEW, rather than trusting a
        # precomputed origin-counts key - fa.analyze_conversation (OLD) has no
        # conditional origin breakdown field at all.
        origins = {"policy_document": 0, "user_utterance": 0, "invented": 0}
        for r in results.values():
            for c in r["tool_calls_log"]:
                for x in c["ids"]:
                    if conditional and not x["pool_nonempty"]:
                        continue
                    if not x["grounded"] and x["origin"]:
                        origins[x["origin"]] += 1
        return tot, ung, origins

    old_flat_tot, old_flat_ung, old_flat_origins = totals(old_results, conditional=False)
    new_flat_tot, new_flat_ung, new_flat_origins = totals(new_results, conditional=False)
    old_cond_tot, old_cond_ung, old_cond_origins = totals(old_results, conditional=True)
    new_cond_tot, new_cond_ung, new_cond_origins = totals(new_results, conditional=True)

    print(f"FLAT       OLD: {old_flat_ung}/{old_flat_tot} ({old_flat_ung/old_flat_tot:.1%}) origins={old_flat_origins}")
    print(f"FLAT       NEW: {new_flat_ung}/{new_flat_tot} ({new_flat_ung/new_flat_tot:.1%}) origins={new_flat_origins}")
    print(f"CONDITIONAL OLD: {old_cond_ung}/{old_cond_tot} ({old_cond_ung/old_cond_tot:.1%}) origins={old_cond_origins}")
    print(f"CONDITIONAL NEW: {new_cond_ung}/{new_cond_tot} ({new_cond_ung/new_cond_tot:.1%}) origins={new_cond_origins}")

    print("\nper-task:")
    for name in llama_sources:
        o, n = old_results[name], new_results[name]
        print(f"  {name}: FLAT old {o['ungrounded_id_arguments']}/{o['total_id_arguments']} "
              f"-> new {n['ungrounded_id_arguments']}/{n['total_id_arguments']}; "
              f"CONDITIONAL old {o['ungrounded_id_arguments_conditional']}/{o['total_id_arguments_conditional']} "
              f"-> new {n['ungrounded_id_arguments_conditional']}/{n['total_id_arguments_conditional']}")

    Path("component_level_grounding_raw.json").write_text(json.dumps(
        {"old": old_results, "new": new_results, "stringified_list_values": all_listish},
        indent=2, default=str))
    print("\nwrote component_level_grounding_raw.json")


if __name__ == "__main__":
    main()
