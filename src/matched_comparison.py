"""Pool failure_analysis.py's per-conversation output over the 3 tasks
both models have (0, 1, 3) - task 2 is Qwen-only. Reads
failure_analysis_raw.json; run failure_analysis.py first to produce it.
"""

from __future__ import annotations

import json
from pathlib import Path

MATCHED_TASKS = {"task0", "task1", "task3"}


def pool(conversations: list[dict]) -> dict:
    total_ids = sum(c["total_id_arguments"] for c in conversations)
    ungrounded = sum(c["ungrounded_id_arguments"] for c in conversations)
    calls_with_id = sum(c["calls_with_any_id"] for c in conversations)
    calls_ungrounded = sum(c["calls_with_ungrounded_id"] for c in conversations)
    origins = {"policy_document": 0, "user_utterance": 0, "invented": 0}
    for c in conversations:
        for k, v in c["origin_counts_ungrounded"].items():
            origins[k] += v

    # CONDITIONAL: restrict to ids/calls made after the grounding pool
    # was already non-empty - the honest rate, per the operator's
    # correction. A conversation whose pool never becomes non-empty
    # (e.g. qwen_task0) contributes 0/0 here, not 0% or 100% - it has
    # NO conditional evidence at all, and must not silently vanish into
    # the denominator as if it were a clean data point.
    total_ids_cond = sum(c["total_id_arguments_conditional"] for c in conversations)
    ungrounded_cond = sum(c["ungrounded_id_arguments_conditional"] for c in conversations)
    calls_with_id_cond = sum(c["calls_with_any_id_conditional"] for c in conversations)
    calls_ungrounded_cond = sum(c["calls_with_ungrounded_id_conditional"] for c in conversations)
    n_tasks_with_conditional_data = sum(1 for c in conversations if c["total_id_arguments_conditional"] > 0)
    n_tasks_pool_never_nonempty = sum(1 for c in conversations if not c["pool_ever_nonempty"])

    return {
        "n_tasks": len(conversations),
        "total_id_arguments": total_ids,
        "ungrounded_id_arguments": ungrounded,
        "ungrounded_fraction_by_argument": (ungrounded / total_ids) if total_ids else None,
        "calls_with_any_id": calls_with_id,
        "calls_with_ungrounded_id": calls_ungrounded,
        "ungrounded_fraction_by_call": (calls_ungrounded / calls_with_id) if calls_with_id else None,
        "origin_counts_ungrounded": origins,
        "n_tasks_with_conditional_data": n_tasks_with_conditional_data,
        "n_tasks_pool_never_nonempty": n_tasks_pool_never_nonempty,
        "total_id_arguments_conditional": total_ids_cond,
        "ungrounded_id_arguments_conditional": ungrounded_cond,
        "ungrounded_fraction_by_argument_conditional": (ungrounded_cond / total_ids_cond) if total_ids_cond else None,
        "calls_with_any_id_conditional": calls_with_id_cond,
        "calls_with_ungrounded_id_conditional": calls_ungrounded_cond,
        "ungrounded_fraction_by_call_conditional": (
            calls_ungrounded_cond / calls_with_id_cond if calls_with_id_cond else None
        ),
    }


def main() -> None:
    raw = json.loads(Path("failure_analysis_raw.json").read_text())

    per_model = {"qwen": {}, "llama": {}}
    for name, data in raw.items():
        model, task = name.split("_", 1)
        if task not in MATCHED_TASKS:
            continue
        per_model[model][task] = data

    result = {}
    for model, tasks in per_model.items():
        missing = MATCHED_TASKS - set(tasks.keys())
        if missing:
            print(f"WARNING: {model} missing matched task(s) {missing} - cannot compute a fair matched pool")
        pooled = pool(list(tasks.values()))
        result[model] = {"per_task": tasks, "pooled": pooled}
        print(f"\n{model} matched (tasks {sorted(tasks.keys())}):")
        print(f"  FLAT by argument: {pooled['ungrounded_id_arguments']}/{pooled['total_id_arguments']} "
              f"({pooled['ungrounded_fraction_by_argument']:.1%})")
        print(f"  FLAT by call: {pooled['calls_with_ungrounded_id']}/{pooled['calls_with_any_id']} "
              f"({pooled['ungrounded_fraction_by_call']:.1%})")
        print(f"  origins (of ungrounded, flat): {pooled['origin_counts_ungrounded']}")
        for task in sorted(tasks.keys()):
            t = tasks[task]
            frac = t["ungrounded_fraction_by_argument"]
            print(f"    {task}: {t['ungrounded_id_arguments']}/{t['total_id_arguments']} "
                  f"({frac:.1%})" if frac is not None else f"    {task}: no ID arguments")

        cfrac = pooled["ungrounded_fraction_by_argument_conditional"]
        print(f"  CONDITIONAL by argument (pool already non-empty at call time - the honest headline): "
              f"{pooled['ungrounded_id_arguments_conditional']}/{pooled['total_id_arguments_conditional']} "
              f"({cfrac:.1%})" if cfrac is not None else "  CONDITIONAL: no data (pool never non-empty in any matched task)")
        ccallfrac = pooled["ungrounded_fraction_by_call_conditional"]
        if ccallfrac is not None:
            print(f"  CONDITIONAL by call: {pooled['calls_with_ungrounded_id_conditional']}/"
                  f"{pooled['calls_with_any_id_conditional']} ({ccallfrac:.1%})")
        print(f"  {pooled['n_tasks_with_conditional_data']}/{pooled['n_tasks']} matched tasks have ANY "
              f"conditional data; {pooled['n_tasks_pool_never_nonempty']} never had a non-empty pool at all")
        for task in sorted(tasks.keys()):
            t = tasks[task]
            cf = t["ungrounded_fraction_by_argument_conditional"]
            if cf is not None:
                print(f"    {task} conditional: {t['ungrounded_id_arguments_conditional']}/"
                      f"{t['total_id_arguments_conditional']} ({cf:.1%})")
            else:
                print(f"    {task} conditional: NO DATA - pool never became non-empty "
                      f"(cascade: first_call_errored={t['cascade']['first_call_errored']}, "
                      f"calls_before_first_success={t['cascade']['calls_before_first_success']})")

    Path("matched_comparison_raw.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nwrote matched_comparison_raw.json")


if __name__ == "__main__":
    main()
