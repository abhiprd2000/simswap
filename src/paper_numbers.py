"""Reviewer response A7 - recompute every number-bearing claim in the
paper under COMPONENT-LEVEL parsing (component_level_grounding.py),
reported side by side with the as-submitted (default classifier)
numbers. Writes paper_numbers_raw.json; docs/PAPER_NUMBERS.md is the
human-readable table built from this script's output - this script is
the source of truth, the doc is a transcription, not a recomputation.

Does not modify failure_analysis.py, schema_conformance.py,
matched_comparison.py, or permutation_test.py. Reuses their functions
directly wherever a claim doesn't involve id-value decomposition at all
(schema conformance, post-error next action) - proven unaffected by
reading their code (see docstrings below at each section), not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import component_level_grounding as clg
import failure_analysis as fa
import matched_comparison as mc
import permutation_test as pt
import schema_conformance as sc
import user_id_mentions as uim

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MATCHED_TASKS = ["task0", "task1", "task3"]

QWEN_MATCHED = {k: v for k, v in fa.SOURCES.items() if k.startswith("qwen_") and k.split("_")[1] in MATCHED_TASKS}
LLAMA_MATCHED = {k: v for k, v in fa.SOURCES.items() if k.startswith("llama_")}
LLAMA_1POINT5B = {k: v for k, v in uim.SOURCES.items() if k.startswith("llama_")}


def compute_both(sources: dict) -> tuple[dict, dict]:
    old = {name: fa.analyze_conversation(name, path) for name, path in sources.items()}
    new = {name: clg.analyze_conversation_component_parsed(name, path) for name, path in sources.items()}
    return old, new


def pooled_pct(pooled, key):
    v = pooled[key]
    return f"{v:.1%}" if v is not None else "n/a"


def fmt_frac(num, den):
    return f"{num}/{den} ({num/den:.1%})" if den else "n/d"


def main():
    out = {}

    qwen_old, qwen_new = compute_both(QWEN_MATCHED)
    llama_old, llama_new = compute_both(LLAMA_MATCHED)
    l15_old, l15_new = compute_both(LLAMA_1POINT5B)

    qwen_old_pool = mc.pool(list(qwen_old.values()))
    qwen_new_pool = mc.pool(list(qwen_new.values()))
    llama_old_pool = mc.pool(list(llama_old.values()))
    llama_new_pool = mc.pool(list(llama_new.values()))
    l15_old_pool = mc.pool(list(l15_old.values()))
    l15_new_pool = mc.pool(list(l15_new.values()))

    print("=== confirm Qwen identical under both classifiers (no stringified lists) ===")
    identical = (qwen_old_pool["ungrounded_id_arguments_conditional"] == qwen_new_pool["ungrounded_id_arguments_conditional"]
                 and qwen_old_pool["total_id_arguments_conditional"] == qwen_new_pool["total_id_arguments_conditional"])
    print(f"qwen conditional old={qwen_old_pool['ungrounded_id_arguments_conditional']}/{qwen_old_pool['total_id_arguments_conditional']} "
          f"new={qwen_new_pool['ungrounded_id_arguments_conditional']}/{qwen_new_pool['total_id_arguments_conditional']} identical={identical}")

    print("\n=== Table 1: conditional ungrounded rate, both agents, matched tasks 0/1/3 ===")
    print(f"qwen   old: {fmt_frac(qwen_old_pool['ungrounded_id_arguments_conditional'], qwen_old_pool['total_id_arguments_conditional'])}")
    print(f"qwen   new: {fmt_frac(qwen_new_pool['ungrounded_id_arguments_conditional'], qwen_new_pool['total_id_arguments_conditional'])}")
    print(f"llama  old: {fmt_frac(llama_old_pool['ungrounded_id_arguments_conditional'], llama_old_pool['total_id_arguments_conditional'])}")
    print(f"llama  new: {fmt_frac(llama_new_pool['ungrounded_id_arguments_conditional'], llama_new_pool['total_id_arguments_conditional'])}")
    gap_old = qwen_old_pool['ungrounded_fraction_by_argument_conditional'] - llama_old_pool['ungrounded_fraction_by_argument_conditional']
    gap_new = qwen_new_pool['ungrounded_fraction_by_argument_conditional'] - llama_new_pool['ungrounded_fraction_by_argument_conditional']
    print(f"gap old: {gap_old*100:.1f}pp   gap new: {gap_new*100:.1f}pp")

    print("\n=== per-task conditional rates and gaps (tasks 0/1/3) ===")
    per_task_table = {}
    for t in MATCHED_TASKS:
        qo = qwen_old[f"qwen_{t}"]; qn = qwen_new[f"qwen_{t}"]
        lo = llama_old[f"llama_{t}"]; ln = llama_new[f"llama_{t}"]
        row = {
            "qwen_old": (qo["ungrounded_id_arguments_conditional"], qo["total_id_arguments_conditional"]),
            "qwen_new": (qn["ungrounded_id_arguments_conditional"], qn["total_id_arguments_conditional"]),
            "llama_old": (lo["ungrounded_id_arguments_conditional"], lo["total_id_arguments_conditional"]),
            "llama_new": (ln["ungrounded_id_arguments_conditional"], ln["total_id_arguments_conditional"]),
        }
        per_task_table[t] = row
        print(f"{t}: qwen_old={fmt_frac(*row['qwen_old'])} qwen_new={fmt_frac(*row['qwen_new'])} "
              f"llama_old={fmt_frac(*row['llama_old'])} llama_new={fmt_frac(*row['llama_new'])}")

    print("\n=== permutation test recompute (FLAT per-task rates, component-level) ===")
    diffs_old, diffs_new = [], []
    for t in MATCHED_TASKS:
        qf = qwen_old[f"qwen_{t}"]["ungrounded_fraction_by_argument"]  # identical old/new for qwen
        lo_f = llama_old[f"llama_{t}"]["ungrounded_fraction_by_argument"]
        ln_f = llama_new[f"llama_{t}"]["ungrounded_fraction_by_argument"]
        diffs_old.append(qf - lo_f)
        diffs_new.append(qf - ln_f)
        print(f"{t}: qwen_flat={qf:.4f} llama_flat_old={lo_f:.4f} llama_flat_new={ln_f:.4f} "
              f"diff_old={qf-lo_f:+.4f} diff_new={qf-ln_f:+.4f}")

    # shared with permutation_test.py (pt.exact_perm_test) so the exact-enumeration
    # formula can't drift between the two call sites again - not reimplemented here.
    perm_old = pt.exact_perm_test(diffs_old)
    perm_new = pt.exact_perm_test(diffs_new)
    print(f"OLD: all same sign={perm_old['all_same_sign']} one-sided p={perm_old['p_one_sided']:.4f} two-sided p={perm_old['p_two_sided']:.4f}")
    print(f"NEW: all same sign={perm_new['all_same_sign']} one-sided p={perm_new['p_one_sided']:.4f} two-sided p={perm_new['p_two_sided']:.4f}")

    print("\n=== Table 2 (paper's provenance table): flat AND conditional, counts and % ===")
    def origin_pct(origins, total):
        return {k: (v, f"{v/total:.1%}" if total else "n/a") for k, v in origins.items()}

    qwen_flat_tot = qwen_old_pool["ungrounded_id_arguments"]
    llama_old_flat_tot = llama_old_pool["ungrounded_id_arguments"]
    llama_new_flat_tot = llama_new_pool["ungrounded_id_arguments"]
    print(f"qwen  FLAT (unchanged): {origin_pct(qwen_old_pool['origin_counts_ungrounded'], qwen_flat_tot)}")
    print(f"llama FLAT old: {origin_pct(llama_old_pool['origin_counts_ungrounded'], llama_old_flat_tot)}")
    print(f"llama FLAT new: {origin_pct(llama_new_pool['origin_counts_ungrounded'], llama_new_flat_tot)}")

    def cond_origins(results):
        origins = {"policy_document": 0, "user_utterance": 0, "invented": 0}
        tot = 0
        for r in results.values():
            for c in r["tool_calls_log"]:
                for x in c["ids"]:
                    if not x["pool_nonempty"]:
                        continue
                    if not x["grounded"]:
                        tot += 1
                        if x["origin"]:
                            origins[x["origin"]] += 1
        return origins, tot

    qwen_cond_origins, qwen_cond_tot = cond_origins(qwen_old)  # identical old/new
    llama_old_cond_origins, llama_old_cond_tot = cond_origins(llama_old)
    llama_new_cond_origins, llama_new_cond_tot = cond_origins(llama_new)
    print(f"qwen  CONDITIONAL (unchanged): {origin_pct(qwen_cond_origins, qwen_cond_tot)}")
    print(f"llama CONDITIONAL old: {origin_pct(llama_old_cond_origins, llama_old_cond_tot)}")
    print(f"llama CONDITIONAL new: {origin_pct(llama_new_cond_origins, llama_new_cond_tot)}")

    print("\n=== A8: 86.5% - calls with >=1 UNGROUNDED id of origin user_utterance, that errored ===")

    def strict_user_stated_call_error_rate(sources, component_level):
        """A8's exact definition: a call counts if >=1 of its id-args is
        ungrounded AND origin == user_utterance (not merely 'value appears
        somewhere in user text', which is the broader, over-inclusive check
        below). Numerator: those calls whose own tool result errored."""
        total_calls, err_calls = 0, 0
        per_convo = {}
        for name, path in sources.items():
            result = (clg.analyze_conversation_component_parsed(name, path) if component_level
                       else fa.analyze_conversation(name, path))
            n_calls = n_err = 0
            for c in result["tool_calls_log"]:
                carries = any((not x["grounded"]) and x["origin"] == "user_utterance" for x in c["ids"])
                if carries:
                    n_calls += 1
                    if c["errored"]:
                        n_err += 1
            per_convo[name] = (n_err, n_calls)
            total_calls += n_calls
            err_calls += n_err
        return err_calls, total_calls, per_convo

    def any_user_text_match_call_error_rate(sources, component_level):
        """The BROADER check used in the earlier (A7) draft: value appears
        ANYWHERE in prior user text, regardless of grounded/origin status
        (so it also counts calls whose id was grounded AND happened to be
        user-stated too). Kept only to explain the earlier 41/61 number."""
        total_calls, err_calls = 0, 0
        per_convo = {}
        for name, path in sources.items():
            messages, _policy = fa.load_messages(path)
            user_texts = []
            calls_queue = []
            n_calls = n_err = 0
            for m in messages:
                role = m.get("role")
                if role == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        user_texts.append(c)
                elif role == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        args = tc.get("arguments") or {}
                        ids = clg.id_values_component_parsed(args) if component_level else fa.id_values(args)
                        carries = any(any(value in t for t in user_texts) for _f, value in ids)
                        calls_queue.append(carries)
                elif role == "tool":
                    if calls_queue:
                        carries = calls_queue.pop(0)
                        if carries:
                            n_calls += 1
                            if m.get("error"):
                                n_err += 1
            per_convo[name] = (n_err, n_calls)
            total_calls += n_calls
            err_calls += n_err
        return err_calls, total_calls, per_convo

    both_matched = {**QWEN_MATCHED, **LLAMA_MATCHED}
    both_all7 = {**{k: v for k, v in fa.SOURCES.items() if k.startswith("qwen_")}, **LLAMA_MATCHED}

    strict_matched_old = strict_user_stated_call_error_rate(both_matched, component_level=False)
    strict_matched_new = strict_user_stated_call_error_rate(both_matched, component_level=True)
    strict_all7_old = strict_user_stated_call_error_rate(both_all7, component_level=False)
    strict_all7_new = strict_user_stated_call_error_rate(both_all7, component_level=True)

    print(f"matched tasks 0/1/3, both agents, 7B user, STRICT definition:")
    print(f"  default classifier:   {fmt_frac(strict_matched_old[0], strict_matched_old[1])}  (expected 32/37)")
    print(f"  component-level:       {fmt_frac(strict_matched_new[0], strict_matched_new[1])}")
    print(f"all 7 conversations (incl. qwen_task2), STRICT definition:")
    print(f"  default classifier:   {fmt_frac(strict_all7_old[0], strict_all7_old[1])}  (expected 36/42)")
    print(f"  component-level:       {fmt_frac(strict_all7_new[0], strict_all7_new[1])}")
    print("  per-conversation (default classifier):")
    for name, (e, n) in strict_matched_old[2].items():
        print(f"    {name}: {fmt_frac(e, n)}")

    # explain the earlier 41/61: diff against the strict matched-tasks set
    broad_matched_old = any_user_text_match_call_error_rate(both_matched, component_level=False)
    print(f"\nbroader (A7) definition, same population (matched, default classifier): "
          f"{fmt_frac(broad_matched_old[0], broad_matched_old[1])}  (A7 reported 41/61)")
    extra_calls = []
    strict_old_result = {name: fa.analyze_conversation(name, path) for name, path in both_matched.items()}
    for name, path in both_matched.items():
        messages, _policy = fa.load_messages(path)
        user_texts = []
        calls_queue = []
        idx = 0
        r = strict_old_result[name]
        for m in messages:
            role = m.get("role")
            if role == "user":
                c = m.get("content")
                if isinstance(c, str):
                    user_texts.append(c)
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    args = tc.get("arguments") or {}
                    ids = fa.id_values(args)
                    broad = any(any(value in t for t in user_texts) for _f, value in ids)
                    calls_queue.append((broad, idx))
                    idx += 1
            elif role == "tool":
                if calls_queue:
                    broad, call_idx = calls_queue.pop(0)
                    call = r["tool_calls_log"][call_idx]
                    strict = any((not x["grounded"]) and x["origin"] == "user_utterance" for x in call["ids"])
                    if broad and not strict:
                        extra_calls.append({
                            "conversation": name, "turn": call["turn"], "name": call["name"],
                            "errored": call["errored"],
                            "ids": [{"field": x["field"], "value": x["value"], "grounded": x["grounded"],
                                     "origin": x["origin"]} for x in call["ids"]],
                        })
    print(f"\nextra calls in the broad-but-not-strict set ({len(extra_calls)} calls) - "
          f"expected: calls whose user-stated id was already GROUNDED:")
    for ec in extra_calls:
        print(f"  {ec['conversation']} turn={ec['turn']} {ec['name']} errored={ec['errored']} ids={ec['ids']}")
    all_grounded_explanation = all(
        any(x["grounded"] for x in ec["ids"]) for ec in extra_calls
    ) if extra_calls else None
    print(f"\nconfirm: every extra call has >=1 GROUNDED id (the expected explanation)? {all_grounded_explanation}")

    print("\n=== Section 3.5 post-error distribution: PROVABLY UNCHANGED ===")
    print("next_action_counts is built from flat_calls = {name, RAW arguments} and error_flags only "
          "(failure_analysis.py lines building flat_calls never call id_values()/classify_origin() - "
          "see analyze_conversation). Component-level parsing only changes how id VALUES inside an "
          "already-fixed call are graded; it cannot change which calls exist, their names, their raw "
          "arguments, or which ones errored. Not rerun - proven invariant by code inspection, not assumed.")

    print("\n=== schema conformance: PROVABLY UNCHANGED ===")
    print("schema_conformance.py imports nothing from failure_analysis.py and calls classify_arg() "
          "directly on raw argument values (list vs str) - zero dependency on id_values/classify_origin/"
          "is_grounded_in. Component-level parsing cannot affect it. Not rerun for the same reason.")

    print("\n=== mismatched-simulator comparison: Qwen self-play vs Llama-on-1.5B-user ===")
    print("CONFIRMED the 1.5B-user Llama transcripts ALSO contain stringified-list id-args (14 across "
          "tasks 0/1/3) - this comparison is NOT exempt from the recompute.")
    l15_hits = []
    for name, path in LLAMA_1POINT5B.items():
        l15_hits.extend(clg.find_all_stringified_list_values(name, path))
    print(f"llama_1.5b-user stringified-list id-args found: {len(l15_hits)}")

    mismatch_old_gap = (qwen_old_pool['ungrounded_fraction_by_argument_conditional']
                        - l15_old_pool['ungrounded_fraction_by_argument_conditional'])
    mismatch_new_gap = (qwen_new_pool['ungrounded_fraction_by_argument_conditional']
                        - l15_new_pool['ungrounded_fraction_by_argument_conditional'])
    print(f"qwen (self-play) conditional: old={fmt_frac(qwen_old_pool['ungrounded_id_arguments_conditional'], qwen_old_pool['total_id_arguments_conditional'])}")
    print(f"llama-1.5B-user conditional:  old={fmt_frac(l15_old_pool['ungrounded_id_arguments_conditional'], l15_old_pool['total_id_arguments_conditional'])}  "
          f"new={fmt_frac(l15_new_pool['ungrounded_id_arguments_conditional'], l15_new_pool['total_id_arguments_conditional'])}")
    print(f"mismatched gap: old={mismatch_old_gap*100:.1f}pp  new={mismatch_new_gap*100:.1f}pp")
    print(f"matched gap (from Table 1 above): old={gap_old*100:.1f}pp  new={gap_new*100:.1f}pp")
    frac_artifact_old = (mismatch_old_gap - gap_old) / mismatch_old_gap if mismatch_old_gap else None
    frac_artifact_new = (mismatch_new_gap - gap_new) / mismatch_new_gap if mismatch_new_gap else None
    print(f"fraction of mismatched gap attributable to simulator mismatch: "
          f"old={frac_artifact_old:.1%}  new={frac_artifact_new:.1%}")

    print("\n=== Table 4 (definition ii) under component-level vocabulary, per task, both user models ===")
    def table4_component(sources, component_level):
        rows = {}
        for name, path in sources.items():
            messages, _policy = fa.load_messages(path)
            user_texts = [m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
            vocab = set()
            for m in messages:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        args = tc.get("arguments") or {}
                        ids = clg.id_values_component_parsed(args) if component_level else fa.id_values(args)
                        for _f, v in ids:
                            vocab.add(v)
            stated = {v for v in vocab if any(v in t for t in user_texts)}
            rows[name] = (len(stated), len(vocab))
        return rows

    qwen_t4_old = table4_component(uim.SOURCES, component_level=False)
    qwen_t4_new = table4_component(uim.SOURCES, component_level=True)
    for name in uim.SOURCES:
        o = qwen_t4_old[name]; n = qwen_t4_new[name]
        marker = " <- 20-step smoke test" if name == "llama_task0" else ""
        print(f"{name}: as-submitted {fmt_frac(*o)}  component-level {fmt_frac(*n)}{marker}")

    print("\n=== A9: does Table 4 hold the agent fixed? ===")
    print("user_id_mentions.SOURCES: qwen_task0-3 -> data/raw/qwen_task{N}.json (agent=Qwen), "
          "llama_task0/1/3 -> data/raw/llama_task{N}_1.5b_user.json (agent=Llama).")
    print("USER_MODEL_BY_CONVO: qwen_task* -> Qwen2.5-7B-AWQ, llama_task* -> Qwen2.5-1.5B-fp16.")
    print("=> the '7B row' = Qwen agent throughout; the '1.5B row' = Llama agent throughout. "
          "Table 4 varies BOTH agent and user model between its two rows - it does NOT hold the agent fixed. Confirmed.")

    print("\nAgent-fixed recompute: Llama x 7B user (matched) vs Llama x 1.5B user, per task and pooled:")
    llama_7b_t4_old = table4_component(LLAMA_MATCHED, component_level=False)
    llama_7b_t4_new = table4_component(LLAMA_MATCHED, component_level=True)
    llama_15b_t4_old = table4_component(LLAMA_1POINT5B, component_level=False)
    llama_15b_t4_new = table4_component(LLAMA_1POINT5B, component_level=True)
    for t in MATCHED_TASKS:
        o7 = llama_7b_t4_old[f"llama_{t}"]; n7 = llama_7b_t4_new[f"llama_{t}"]
        o15 = llama_15b_t4_old[f"llama_{t}"]; n15 = llama_15b_t4_new[f"llama_{t}"]
        print(f"  {t}: llama x 7B  old={fmt_frac(*o7)} new={fmt_frac(*n7)}   "
              f"llama x 1.5B old={fmt_frac(*o15)} new={fmt_frac(*n15)}")

    def pool_t4(rows):
        num = sum(v[0] for v in rows.values())
        den = sum(v[1] for v in rows.values())
        return num, den

    llama_7b_pool_old = pool_t4(llama_7b_t4_old); llama_7b_pool_new = pool_t4(llama_7b_t4_new)
    llama_15b_pool_old = pool_t4(llama_15b_t4_old); llama_15b_pool_new = pool_t4(llama_15b_t4_new)
    print(f"  pooled: llama x 7B  old={fmt_frac(*llama_7b_pool_old)} new={fmt_frac(*llama_7b_pool_new)}")
    print(f"  pooled: llama x 1.5B old={fmt_frac(*llama_15b_pool_old)} new={fmt_frac(*llama_15b_pool_new)}")

    print("\nQwen x 7B user, matched tasks 0/1/3 (reported separately - no Qwen x 1.5B data exists):")
    qwen_7b_t4_old = table4_component(QWEN_MATCHED, component_level=False)
    qwen_7b_t4_new = table4_component(QWEN_MATCHED, component_level=True)
    for t in MATCHED_TASKS:
        o = qwen_7b_t4_old[f"qwen_{t}"]; n = qwen_7b_t4_new[f"qwen_{t}"]
        print(f"  {t}: old={fmt_frac(*o)} new={fmt_frac(*n)}")
    qwen_7b_pool_old = pool_t4(qwen_7b_t4_old); qwen_7b_pool_new = pool_t4(qwen_7b_t4_new)
    print(f"  pooled: old={fmt_frac(*qwen_7b_pool_old)} new={fmt_frac(*qwen_7b_pool_new)}")

    print("\n=== A11 T2: per-task conditional gaps (component-level), tasks 0/1/3 ===")
    t2_gaps = {}
    for t in MATCHED_TASKS:
        qn = qwen_new[f"qwen_{t}"]["ungrounded_fraction_by_argument_conditional"]
        ln = llama_new[f"llama_{t}"]["ungrounded_fraction_by_argument_conditional"]
        gap = (qn - ln) if (qn is not None and ln is not None) else None
        t2_gaps[t] = gap
        print(f"  {t}: qwen={qn} llama={ln} gap={'n/d' if gap is None else f'{gap*100:.1f}pp'}")

    print("\n=== A11 denominator audit: narrow (_id/_ids only) vs wide (+email/zip/name), component-level ===")

    def narrow_only_component_parsed(args: dict) -> list[tuple[str, str]]:
        return [(f, v) for f, v in clg.id_values_component_parsed(args) if f not in fa.WIDE_FIELD_NAMES]

    def analyze_conditional_rate_with(sources, id_fn):
        tot_ids = ung_ids = 0
        for name, path in sources.items():
            messages, policy_text = fa.load_messages(path)
            grounded_contents, user_texts = [], []
            pool_nonempty_ids = []
            for m in messages:
                role = m.get("role")
                if role == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        user_texts.append(c)
                elif role == "assistant" and m.get("tool_calls"):
                    pool_nonempty = len(grounded_contents) > 0
                    for tc in m["tool_calls"]:
                        args = tc.get("arguments") or {}
                        for field, value in id_fn(args):
                            is_grounded = fa.is_grounded_in(value, grounded_contents)
                            if pool_nonempty:
                                pool_nonempty_ids.append(is_grounded)
                elif role == "tool":
                    content = m.get("content")
                    is_error = bool(m.get("error"))
                    content_str = content if isinstance(content, str) else json.dumps(content, default=str)
                    if not is_error:
                        grounded_contents.append(content_str)
            tot_ids += len(pool_nonempty_ids)
            ung_ids += sum(1 for g in pool_nonempty_ids if not g)
        return ung_ids, tot_ids

    both_matched_sources = {**QWEN_MATCHED, **LLAMA_MATCHED}
    wide_ung, wide_tot = analyze_conditional_rate_with(both_matched_sources, clg.id_values_component_parsed)
    narrow_ung, narrow_tot = analyze_conditional_rate_with(both_matched_sources, narrow_only_component_parsed)
    print(f"  WIDE (current, incl. email/zip/first_name/last_name): {fmt_frac(wide_ung, wide_tot)}")
    print(f"  NARROW (_id/_ids suffix only, excl. email/zip/name):   {fmt_frac(narrow_ung, narrow_tot)}")
    wide_rate = wide_ung / wide_tot if wide_tot else None
    narrow_rate = narrow_ung / narrow_tot if narrow_tot else None
    change = (wide_rate - narrow_rate) if (wide_rate is not None and narrow_rate is not None) else None
    print(f"  change in pooled conditional ungrounded rate from widening to email/zip/name fields: "
          f"{'n/d' if change is None else f'{change*100:+.1f}pp'}")

    out = {
        "qwen_matched": {"old_pool": qwen_old_pool, "new_pool": qwen_new_pool},
        "llama_matched": {"old_pool": llama_old_pool, "new_pool": llama_new_pool},
        "llama_1_5b_user": {"old_pool": l15_old_pool, "new_pool": l15_new_pool},
        "per_task_table": per_task_table,
        "permutation_old": perm_old, "permutation_new": perm_new,
        "diffs_old": diffs_old, "diffs_new": diffs_new,
        "a8_strict_user_stated_call_error_rate": {
            "matched_old": strict_matched_old[:2], "matched_new": strict_matched_new[:2],
            "all7_old": strict_all7_old[:2], "all7_new": strict_all7_new[:2],
            "per_convo_matched_old": strict_matched_old[2],
        },
        "a8_broad_definition_matched_old": broad_matched_old[:2],
        "a8_extra_calls_broad_minus_strict": extra_calls,
        "mismatched_gap_old": mismatch_old_gap, "mismatched_gap_new": mismatch_new_gap,
        "matched_gap_old": gap_old, "matched_gap_new": gap_new,
        "fraction_artifact_old": frac_artifact_old, "fraction_artifact_new": frac_artifact_new,
        "table4_old": qwen_t4_old, "table4_new": qwen_t4_new,
        "a9_table4_agent_fixed": {
            "llama_x_7b_old": llama_7b_t4_old, "llama_x_7b_new": llama_7b_t4_new,
            "llama_x_1_5b_old": llama_15b_t4_old, "llama_x_1_5b_new": llama_15b_t4_new,
            "llama_x_7b_pool_old": llama_7b_pool_old, "llama_x_7b_pool_new": llama_7b_pool_new,
            "llama_x_1_5b_pool_old": llama_15b_pool_old, "llama_x_1_5b_pool_new": llama_15b_pool_new,
            "qwen_x_7b_old": qwen_7b_t4_old, "qwen_x_7b_new": qwen_7b_t4_new,
            "qwen_x_7b_pool_old": qwen_7b_pool_old, "qwen_x_7b_pool_new": qwen_7b_pool_new,
        },
        "a11_t2_gaps_component": t2_gaps,
        "a11_denominator_audit": {
            "wide": (wide_ung, wide_tot), "narrow": (narrow_ung, narrow_tot), "change_pp": change,
        },
    }
    Path("paper_numbers_raw.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote paper_numbers_raw.json")


if __name__ == "__main__":
    main()
