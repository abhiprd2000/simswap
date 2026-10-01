"""Grounding-rate, provenance, and post-error-recovery metrics for tau2 transcripts.

An ID-shaped tool-call argument (field name ending `_id`/`_ids`, or one of
`email`/`zip`/`first_name`/`last_name`) is grounded if its exact value appears as a
substring of an earlier successful tool result in the same conversation, checked
causally. See docs/FAILURE_ANALYSIS.md for the full method writeup, the two analysis
bugs found and fixed while building this (docs/AUDIT_REPORT.md), and why both a FLAT
and a CONDITIONAL (pool-already-nonempty) rate are reported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCES = {
    "qwen_task0": DATA_DIR / "qwen_task0.json",
    "qwen_task1": DATA_DIR / "qwen_task1.json",
    "qwen_task2": DATA_DIR / "qwen_task2.json",
    "qwen_task3": DATA_DIR / "qwen_task3.json",
    "llama_task0": DATA_DIR / "llama_task0.json",
    "llama_task1": DATA_DIR / "llama_task1.json",
    "llama_task3": DATA_DIR / "llama_task3.json",
    # llama_task2 has no transcript: vLLM's llama3_json parser hard-rejects the
    # genuine parallel tool-call attempt it makes there. See docs/SCHEMA_CONFORMANCE.md.
}

NARROW_FIELD_RE = re.compile(r"_id$|_ids$")
WIDE_FIELD_NAMES = {"email", "zip", "first_name", "last_name"}

LOOKUP_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details",
                "get_user_details", "list_all_product_types"}


def load_messages(path) -> tuple[list[dict], str]:
    d = json.loads(Path(path).read_text())
    sim = d["sim_dump"]
    return sim["messages"], sim.get("policy", "") or ""


def id_values(args: dict) -> list[tuple[str, str]]:
    """(field_name, value) pairs for arguments that look like identifiers."""
    out = []
    for k, v in args.items():
        is_id_field = bool(NARROW_FIELD_RE.search(k)) or (k in WIDE_FIELD_NAMES)
        if not is_id_field:
            continue
        if isinstance(v, str):
            out.append((k, v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    out.append((k, item))
    return out


def classify_origin(value: str, policy_text: str, user_texts: list[str]) -> str:
    if value in policy_text:
        return "policy_document"
    for t in user_texts:
        if value in t:
            return "user_utterance"
    return "invented"


def is_grounded_in(value: str, contents: list[str]) -> bool:
    """True if value appears as a substring of any prior successful result."""
    return any(value in c for c in contents)


def analyze_conversation(name: str, path) -> dict:
    messages, policy_text = load_messages(path)

    grounded_contents: list[str] = []
    user_texts: list[str] = []
    tool_calls_log = []
    pending_by_field: dict[str, list[str]] = {}
    post_error_actions = []
    result_queue: list[tuple[str, str, int]] = []
    call_result_queue: list[int] = []
    flat_calls: list[dict] = []  # (name, arguments) in call order, for next-action classification
    error_flags: list[bool] = []  # parallel to tool results, in order

    for i, m in enumerate(messages):
        role = m.get("role")

        if role == "user":
            c = m.get("content")
            if isinstance(c, str):
                user_texts.append(c)

        elif role == "assistant" and m.get("tool_calls"):
            # pool state BEFORE this turn's call(s): a flat rate over a whole
            # conversation conflates "agent ignored something it had" with "agent
            # had nothing to check yet" - tag every id/call with this so the
            # conditional rate below can separate the two.
            pool_nonempty = len(grounded_contents) > 0
            for tc in m["tool_calls"]:
                args = tc.get("arguments") or {}
                flat_calls.append({"name": tc.get("name"), "arguments": args})
                ids = id_values(args)
                id_records = []
                for field, value in ids:
                    is_grounded = is_grounded_in(value, grounded_contents)
                    origin = None if is_grounded else classify_origin(value, policy_text, user_texts)
                    id_records.append({
                        "field": field, "value": value, "grounded": is_grounded,
                        "origin": origin, "pool_nonempty": pool_nonempty,
                    })

                    if field in pending_by_field:
                        prev_values = pending_by_field[field]
                        action = "retry_same_id" if value in prev_values else "invent_new_id"
                        post_error_actions.append({
                            "field": field, "prev_values": list(prev_values),
                            "new_value": value, "action": action,
                        })
                    result_queue.append((field, value, len(tool_calls_log)))

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
            error_flags.append(is_error)

            if call_result_queue:
                call_idx = call_result_queue.pop(0)
                tool_calls_log[call_idx]["errored"] = is_error

            # id-tracking queue stays in sync for pending_by_field bookkeeping, but
            # the grounding-pool update below is unconditional on any successful
            # result (v1 bug fix, see docs/AUDIT_REPORT.md).
            if result_queue:
                field, value, _call_i = result_queue.pop(0)
                if is_error:
                    pending_by_field.setdefault(field, [])
                    if value not in pending_by_field[field]:
                        pending_by_field[field].append(value)
                else:
                    pending_by_field.pop(field, None)

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

    action_counts: dict[str, int] = {}
    for a in post_error_actions:
        action_counts[a["action"]] = action_counts.get(a["action"], 0) + 1

    # CONDITIONAL rate: restrict to ids/calls made after the pool already held
    # >=1 success - the honest headline. The flat rate is inflated by calls
    # where grounding was never possible in the first place.
    ids_pool_nonempty = [x for c in tool_calls_log for x in c["ids"] if x["pool_nonempty"]]
    ungrounded_pool_nonempty = sum(1 for x in ids_pool_nonempty if not x["grounded"])
    calls_pool_nonempty = [c for c in tool_calls_log if c["pool_nonempty"] and c["ids"]]
    calls_ungrounded_pool_nonempty = sum(1 for c in calls_pool_nonempty if any(not x["grounded"] for x in c["ids"]))

    # CASCADE: is the flat rate arithmetic fallout from a bad entry point (first
    # call ungrounded -> errors -> pool stays empty -> everything after is
    # forced-ungrounded), or a genuinely flat failure rate?
    first_id_bearing_call = next((c for c in tool_calls_log if c["ids"]), None)
    cascade = {
        "pool_ever_nonempty": len(grounded_contents) > 0,
        "first_call_ungrounded": (any(not x["grounded"] for x in first_id_bearing_call["ids"])
                                   if first_id_bearing_call else None),
        "first_call_errored": first_id_bearing_call["errored"] if first_id_bearing_call else None,
        "calls_before_first_success": sum(1 for c in tool_calls_log if not c["pool_nonempty"]),
    }

    # POST-ERROR NEXT ACTION: for each tool-call error, classify the very next
    # call: a genuine lookup/verification tool, an exact repeat, the same tool
    # with different (still unverified) args, something else, or no next call.
    next_action_counts = {"exact_same_call": 0, "same_tool_diff_args": 0, "lookup_tool": 0,
                           "other": 0, "no_next_call": 0}
    for i, was_error in enumerate(error_flags):
        if not was_error:
            continue
        if i + 1 >= len(flat_calls):
            next_action_counts["no_next_call"] += 1
            continue
        cur, nxt = flat_calls[i], flat_calls[i + 1]
        if nxt["name"] in LOOKUP_TOOLS:
            next_action_counts["lookup_tool"] += 1
        elif nxt["name"] == cur["name"] and nxt["arguments"] == cur["arguments"]:
            next_action_counts["exact_same_call"] += 1
        elif nxt["name"] == cur["name"]:
            next_action_counts["same_tool_diff_args"] += 1
        else:
            next_action_counts["other"] += 1

    return {
        "name": name,
        "total_tool_calls": len(tool_calls_log),
        "total_id_arguments": total_ids,
        "ungrounded_id_arguments": ungrounded_ids,
        "ungrounded_fraction_by_argument": (ungrounded_ids / total_ids) if total_ids else None,
        "calls_with_any_id": calls_with_id,
        "calls_with_ungrounded_id": calls_with_ungrounded,
        "ungrounded_fraction_by_call": (calls_with_ungrounded / calls_with_id) if calls_with_id else None,
        "origin_counts_ungrounded": origin_counts,
        "post_error_action_counts": action_counts,
        "tool_calls_log": tool_calls_log,
        "post_error_actions": post_error_actions,
        "pool_ever_nonempty": cascade["pool_ever_nonempty"],
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
        "cascade": cascade,
        "next_action_counts": next_action_counts,
    }


def main() -> None:
    results = {}
    next_action_total = {"exact_same_call": 0, "same_tool_diff_args": 0, "lookup_tool": 0,
                          "other": 0, "no_next_call": 0}
    for name, path in SOURCES.items():
        if not Path(path).exists():
            print(f"MISSING: {name} -> {path}")
            continue
        results[name] = analyze_conversation(name, path)
        r = results[name]
        print(f"{name}: FLAT {r['ungrounded_id_arguments']}/{r['total_id_arguments']} ungrounded id-args "
              f"({r['ungrounded_fraction_by_argument']}), "
              f"CONDITIONAL {r['ungrounded_id_arguments_conditional']}/{r['total_id_arguments_conditional']} "
              f"({r['ungrounded_fraction_by_argument_conditional']}), origins={r['origin_counts_ungrounded']}")
        for k in next_action_total:
            next_action_total[k] += r["next_action_counts"][k]

    print(f"\npost-error next action, pooled: {next_action_total}")

    Path("failure_analysis_raw.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nwrote failure_analysis_raw.json")


if __name__ == "__main__":
    main()
