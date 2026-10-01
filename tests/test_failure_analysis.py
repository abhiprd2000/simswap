"""Regression tests for failure_analysis.py's grounding logic.

Runs the REAL production function (analyze_conversation) against small
hand-built fixture transcripts, not a reimplementation - a bug in
failure_analysis.py should make one of these fail. No pytest
dependency: plain asserts, a main() that runs all cases and reports
pass/fail, non-zero exit on any failure.

Written 2026-08-16 after a real bug was found and fixed (see
AUDIT_REPORT.md): the grounding pool was only updated when the
resolved call itself had a tracked _id-suffixed argument, so a
successful find_user_id_by_name_zip result never grounded a later
get_user_details call. Test 1 below locks that exact case down.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import failure_analysis as fa
import matched_comparison as mc
import schema_conformance as sc

FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="simswap_regtest_"))


def make_fixture(name: str, messages: list[dict], policy: str = "") -> str:
    path = FIXTURE_DIR / f"{name}.json"
    path.write_text(json.dumps({"sim_dump": {"messages": messages, "policy": policy}}))
    return str(path)


def tool_call(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "tool_calls": [{"id": "x", "name": name, "arguments": arguments, "requestor": "assistant"}]}


def tool_result(content, error: bool = False) -> dict:
    return {"role": "tool", "content": content, "error": error, "requestor": "assistant"}


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def get_call_ids(result: dict, call_index: int) -> list[dict]:
    return result["tool_calls_log"][call_index]["ids"]


def find_id(ids: list[dict], field: str, value: str) -> dict:
    for x in ids:
        if x["field"] == field and x["value"] == value:
            return x
    raise AssertionError(f"no id record for {field}={value!r} in {ids}")


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_grounded_via_non_suffixed_arg_tool():
    """THE BUG REGRESSION TEST - must never regress.

    find_user_id_by_name_zip's own arguments (first_name/last_name/zip)
    have no _id suffix. Its successful result must still ground a
    later get_user_details(user_id=...) call using that exact value.
    """
    messages = [
        tool_call("find_user_id_by_name_zip", {"first_name": "Jane", "last_name": "Doe", "zip": "12345"}),
        tool_result("user_abc123", error=False),
        tool_call("get_user_details", {"user_id": "user_abc123"}),
        tool_result('{"user_id": "user_abc123", "name": "Jane Doe"}', error=False),
    ]
    path = make_fixture("bug_regression", messages)
    result = fa.analyze_conversation("bug_regression", path)
    rec = find_id(get_call_ids(result, 1), "user_id", "user_abc123")
    assert rec["grounded"] is True, f"REGRESSION: the fixed bug is back. {rec}"


@case
def test_id_only_in_error_result_stays_ungrounded():
    """An ID that only ever appears in ERROR result content (echoed
    back) must not count as grounded."""
    messages = [
        tool_call("get_item_details", {"item_id": "FAKE-999"}),
        tool_result("Error: item FAKE-999 not found", error=True),
        tool_call("get_item_details", {"item_id": "FAKE-999"}),
        tool_result("Error: item FAKE-999 not found", error=True),
    ]
    path = make_fixture("error_echo", messages)
    result = fa.analyze_conversation("error_echo", path)
    for i in (0, 1):
        rec = find_id(get_call_ids(result, i), "item_id", "FAKE-999")
        assert rec["grounded"] is False, f"error-echo wrongly counted as grounding: {rec}"


@case
def test_causal_ordering_future_result_does_not_ground_past_use():
    """An ID used BEFORE it's ever confirmed by a successful result
    must count as ungrounded at the time of use, even if that same ID
    shows up in a LATER successful result (ordering violation check)."""
    messages = [
        tool_call("get_item_details", {"item_id": "ITEM-777"}),
        tool_result("Error: not found", error=True),
        tool_call("get_order_details", {"order_id": "#O1"}),
        tool_result('{"order_id": "#O1", "items": ["ITEM-777"]}', error=False),
    ]
    path = make_fixture("causal_ordering", messages)
    result = fa.analyze_conversation("causal_ordering", path)
    first_use = find_id(get_call_ids(result, 0), "item_id", "ITEM-777")
    assert first_use["grounded"] is False, (
        "causal ordering violated: an ID was marked grounded using a result "
        f"that came AFTER its use. {first_use}"
    )


@case
def test_legitimate_grounding_same_and_later_turns():
    """A truly grounded ID must be recognized both immediately after
    its grounding result (adjacent turn) and much later (non-adjacent
    turn) - and an unrelated argument in that same later call that was
    never grounded must still show up as ungrounded."""
    messages = [
        tool_call("get_order_details", {"order_id": "#O2"}),
        tool_result('{"order_id": "#O2", "items": ["ITEM-555"]}', error=False),
        tool_call("get_item_details", {"item_id": "ITEM-555"}),  # adjacent turn
        tool_result('{"item_id": "ITEM-555", "name": "Widget"}', error=False),
        tool_call("exchange_delivered_order_items", {           # later turn, reuses order_id/item
            "order_id": "#O2", "item_ids": ["ITEM-555"],
            "new_item_ids": ["ITEM-555"], "payment_method_id": "pm_never_seen",
        }),
        tool_result("Error: payment method not found", error=True),
    ]
    path = make_fixture("legit_grounding", messages)
    result = fa.analyze_conversation("legit_grounding", path)

    adjacent = find_id(get_call_ids(result, 1), "item_id", "ITEM-555")
    assert adjacent["grounded"] is True, f"adjacent-turn grounding failed: {adjacent}"

    later_order = find_id(get_call_ids(result, 2), "order_id", "#O2")
    later_item = find_id(get_call_ids(result, 2), "item_ids", "ITEM-555")
    assert later_order["grounded"] is True, f"later-turn order_id grounding failed: {later_order}"
    assert later_item["grounded"] is True, f"later-turn item_ids grounding failed: {later_item}"

    never_grounded = find_id(get_call_ids(result, 2), "payment_method_id", "pm_never_seen")
    assert never_grounded["grounded"] is False, f"invented payment_method_id wrongly grounded: {never_grounded}"


@case
def test_non_id_suffixed_arguments_are_in_the_denominator():
    """email/zip/first_name/last_name must be picked up as ID-shaped
    arguments (the AUDIT 1 denominator fix) - not silently skipped."""
    messages = [
        tool_call("find_user_id_by_email", {"email": "test@example.com"}),
        tool_result("Error: not found", error=True),
        tool_call("find_user_id_by_name_zip", {"first_name": "A", "last_name": "B", "zip": "00000"}),
        tool_result("Error: not found", error=True),
    ]
    path = make_fixture("wide_fields", messages)
    result = fa.analyze_conversation("wide_fields", path)

    email_ids = [x for c in result["tool_calls_log"] for x in c["ids"] if x["field"] == "email"]
    assert len(email_ids) == 1, f"email argument not picked up as ID-shaped: {result['tool_calls_log']}"
    assert email_ids[0]["grounded"] is False

    for field in ("first_name", "last_name", "zip"):
        matches = [x for c in result["tool_calls_log"] for x in c["ids"] if x["field"] == field]
        assert len(matches) == 1, f"{field} argument not picked up as ID-shaped: {result['tool_calls_log']}"
        assert matches[0]["grounded"] is False


@case
def test_all_error_conversation_is_fully_ungrounded():
    """Locks in the qwen_task0 pattern, hand-verified 2026-08-16: a
    conversation with ZERO successful tool results must show every
    single ID argument as ungrounded, with no false positives from an
    empty grounding pool ever accidentally matching (e.g. an empty
    string or None being treated as "in" some default pool)."""
    messages = [
        tool_call("exchange_delivered_order_items", {
            "order_id": "#W1", "item_ids": ["A-1"], "new_item_ids": ["A-1"], "payment_method_id": "pm_1",
        }),
        tool_result("Error: item not found", error=True),
        tool_call("get_item_details", {"item_id": "A-1"}),
        tool_result("Error: item not found", error=True),
        tool_call("get_product_details", {"product_id": "P-1"}),
        tool_result("Error: product not found", error=True),
    ]
    path = make_fixture("all_error", messages)
    result = fa.analyze_conversation("all_error", path)
    assert result["ungrounded_id_arguments"] == result["total_id_arguments"] > 0, (
        f"all-error conversation should be 100% ungrounded, got {result['ungrounded_id_arguments']}"
        f"/{result['total_id_arguments']}"
    )
    for c in result["tool_calls_log"]:
        for x in c["ids"]:
            assert x["grounded"] is False, f"false-positive grounding with zero successful results: {x}"


@case
def test_schema_conformance_detects_str_instead_of_list():
    """Second failure axis (2026-08-16, operator instruction): item_ids/
    new_item_ids must be a real list. A string that merely LOOKS like a
    list (e.g. "[1151293680]", the real llama_task0 case) must be
    flagged - and a genuinely well-formed call must NOT be flagged."""
    malformed = sc.classify_arg("item_ids", "[1151293680]")
    assert malformed is not None and "str-instead-of-list" in malformed, (
        f"schema conformance check failed to catch a string-typed item_ids: {malformed}"
    )

    well_formed = sc.classify_arg("item_ids", ["1151293680"])
    assert well_formed is None, f"well-formed list wrongly flagged: {well_formed}"

    non_list_field = sc.classify_arg("order_id", "#W1234567")
    assert non_list_field is None, f"a normal str field on a str value wrongly flagged: {non_list_field}"

    wrong_direction = sc.classify_arg("order_id", ["#W1234567"])
    assert wrong_direction is not None and "list-instead-of-str" in wrong_direction, (
        f"a list where str is expected should also be flagged: {wrong_direction}"
    )


@case
def test_empty_pool_conversation_has_no_conditional_evidence():
    """THE OPERATOR'S CORRECTION - must never regress. A conversation
    whose grounding pool is NEVER non-empty (every tool result is an
    error, e.g. qwen_task0) has ALL ids ungrounded by construction, not
    by agent behavior - the FLAT rate is 100% but that is arithmetically
    forced, not evidence. The CONDITIONAL metrics (restricted to calls
    made after the pool was already non-empty) must reflect NO evidence
    at all: 0/0, None fraction - never silently 0% or 100%. The cascade
    fields must also show every call happened before any success."""
    messages = [
        tool_call("get_item_details", {"item_id": "A-1"}),
        tool_result("Error: not found", error=True),
        tool_call("get_item_details", {"item_id": "B-2"}),
        tool_result("Error: not found", error=True),
        tool_call("get_item_details", {"item_id": "C-3"}),
        tool_result("Error: not found", error=True),
    ]
    path = make_fixture("empty_pool", messages)
    result = fa.analyze_conversation("empty_pool", path)

    assert result["pool_ever_nonempty"] is False, "pool must be reported as never non-empty"
    assert result["ungrounded_fraction_by_argument"] == 1.0, "flat rate is still 100% - forced, not wrong"
    assert result["total_id_arguments_conditional"] == 0, (
        f"REGRESSION: a conversation with no successful result must contribute ZERO conditional "
        f"denominator, not silently count as evidence: {result['total_id_arguments_conditional']}"
    )
    assert result["ungrounded_id_arguments_conditional"] == 0
    assert result["ungrounded_fraction_by_argument_conditional"] is None, (
        "conditional fraction must be None (no evidence), not 0.0 or 1.0, when the pool never fills"
    )
    assert result["calls_with_any_id_conditional"] == 0
    assert result["ungrounded_fraction_by_call_conditional"] is None

    c = result["cascade"]
    assert c["pool_ever_nonempty"] is False
    assert c["first_call_ungrounded"] is True
    assert c["first_call_errored"] is True
    assert c["calls_before_first_success"] == 3, (
        f"all 3 calls happened before any success (there was none) - got {c['calls_before_first_success']}"
    )


@case
def test_conditional_rate_excludes_only_pre_success_calls():
    """A conversation where the pool becomes non-empty PARTWAY through:
    the conditional metrics must include only ids/calls from AFTER that
    point, and the flat and conditional rates must genuinely differ
    (otherwise this test isn't checking anything real)."""
    messages = [
        tool_call("get_item_details", {"item_id": "X-1"}),          # before any success - excluded from conditional
        tool_result("Error: not found", error=True),
        tool_call("get_order_details", {"order_id": "#O9"}),        # this call SUCCEEDS
        tool_result('{"order_id": "#O9", "items": ["Y-2"]}', error=False),
        tool_call("get_item_details", {"item_id": "Y-2"}),          # after success, and IS grounded
        tool_result('{"item_id": "Y-2"}', error=False),
        tool_call("get_item_details", {"item_id": "Z-3"}),          # after success, but NOT grounded
        tool_result("Error: not found", error=True),
    ]
    path = make_fixture("partial_pool", messages)
    result = fa.analyze_conversation("partial_pool", path)

    assert result["pool_ever_nonempty"] is True
    # flat: X-1 ungrounded, order_id #O9 ungrounded (first success, nothing
    # grounds it), Y-2 grounded, Z-3 ungrounded -> 3 ungrounded / 4 total
    assert result["total_id_arguments"] == 4
    assert result["ungrounded_id_arguments"] == 3
    # conditional: only ids on calls made AFTER the pool had >=1 success -
    # that's Y-2 (grounded) and Z-3 (ungrounded) = 2 total, 1 ungrounded.
    # X-1 (before) and #O9 (the call that CREATES the first success) are
    # excluded - #O9 was issued while the pool was still empty.
    assert result["total_id_arguments_conditional"] == 2, (
        f"expected 2 conditional ids (Y-2, Z-3), got {result['total_id_arguments_conditional']}"
    )
    assert result["ungrounded_id_arguments_conditional"] == 1, (
        f"expected 1 ungrounded conditional id (Z-3 only, Y-2 is grounded), "
        f"got {result['ungrounded_id_arguments_conditional']}"
    )
    assert result["ungrounded_fraction_by_argument_conditional"] == 0.5
    assert result["ungrounded_fraction_by_argument_conditional"] != result["ungrounded_fraction_by_argument"], (
        "conditional and flat rates must genuinely differ in this fixture - "
        "if they're equal, the conditional restriction isn't doing anything"
    )

    c = result["cascade"]
    assert c["calls_before_first_success"] == 2, "the item_id call and the order_id call that succeeds"


@case
def test_matched_pool_aggregates_correctly():
    """matched_comparison.pool() must sum raw counts (not average
    fractions) and recompute the fraction from the summed totals -
    averaging per-task fractions directly would misweight tasks with
    different denominators. Uses hand-built per-conversation dicts
    shaped like failure_analysis_raw.json's entries, not real files."""
    fake_conversations = [
        {"total_id_arguments": 10, "ungrounded_id_arguments": 10,
         "calls_with_any_id": 5, "calls_with_ungrounded_id": 5,
         "origin_counts_ungrounded": {"policy_document": 0, "user_utterance": 7, "invented": 3},
         "pool_ever_nonempty": False,
         "total_id_arguments_conditional": 0, "ungrounded_id_arguments_conditional": 0,
         "calls_with_any_id_conditional": 0, "calls_with_ungrounded_id_conditional": 0,
         "ungrounded_fraction_by_argument_conditional": None,
         "cascade": {"first_call_errored": True, "calls_before_first_success": 5}},
        {"total_id_arguments": 20, "ungrounded_id_arguments": 5,
         "calls_with_any_id": 8, "calls_with_ungrounded_id": 2,
         "origin_counts_ungrounded": {"policy_document": 0, "user_utterance": 1, "invented": 4},
         "pool_ever_nonempty": True,
         "total_id_arguments_conditional": 12, "ungrounded_id_arguments_conditional": 3,
         "calls_with_any_id_conditional": 5, "calls_with_ungrounded_id_conditional": 1,
         "ungrounded_fraction_by_argument_conditional": 0.25,
         "cascade": {"first_call_errored": False, "calls_before_first_success": 1}},
    ]
    pooled = mc.pool(fake_conversations)
    assert pooled["total_id_arguments"] == 30
    assert pooled["ungrounded_id_arguments"] == 15
    # 15/30 = 0.5 exactly - NOT the naive average of the two per-task
    # fractions (10/10=1.0, 5/20=0.25 -> naive average 0.625, wrong).
    assert pooled["ungrounded_fraction_by_argument"] == 0.5, (
        f"pooled fraction must be computed from summed counts, not averaged "
        f"per-task fractions: got {pooled['ungrounded_fraction_by_argument']}"
    )
    assert pooled["calls_with_any_id"] == 13
    assert pooled["calls_with_ungrounded_id"] == 7
    assert pooled["origin_counts_ungrounded"] == {"policy_document": 0, "user_utterance": 8, "invented": 7}
    # conditional: only the second fake conversation contributes (the
    # first's pool never fills) - must sum correctly and NOT let the
    # first conversation's 0/0 silently corrupt the pooled fraction.
    assert pooled["total_id_arguments_conditional"] == 12
    assert pooled["ungrounded_id_arguments_conditional"] == 3
    assert pooled["ungrounded_fraction_by_argument_conditional"] == 0.25
    assert pooled["n_tasks_with_conditional_data"] == 1
    assert pooled["n_tasks_pool_never_nonempty"] == 1


@case
def test_schema_conformance_records_llama_parallel_tool_call_rejection():
    """The second schema-conformance axis (request-structure rejection,
    not argument-type) must keep llama_task2's parallel-tool-call
    rejection registered - it can't be derived from a transcript (none
    exists for that attempt), so nothing else guards against it being
    silently lost in a future edit. Must NOT appear for qwen."""
    llama_events = [e for e in sc.NON_TRANSCRIPT_EVENTS if e["model"] == "llama"]
    assert any(e["conversation"] == "llama_task2" and e["malformation"] == "parallel_tool_calls_rejected"
               for e in llama_events), (
        f"llama_task2's parallel-tool-call rejection is missing from NON_TRANSCRIPT_EVENTS: {sc.NON_TRANSCRIPT_EVENTS}"
    )
    qwen_events = [e for e in sc.NON_TRANSCRIPT_EVENTS if e["model"] == "qwen"]
    assert qwen_events == [], f"qwen should have no request-structure rejections recorded: {qwen_events}"


def main() -> None:
    failures = []
    for fn in CASES:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"FAIL: {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(fn.__name__)
            print(f"ERROR: {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} passed")
    if failures:
        print("FAILED:", failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
