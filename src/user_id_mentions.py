"""Mechanism check: how many ID-shaped values does the USER side state per
conversation, by user model? Deliberately compares two DIFFERENT user
models (Qwen2.5-7B-Instruct-AWQ vs. the smaller Qwen2.5-1.5B-Instruct used
in an earlier, superseded Llama run) - that contrast is the whole point,
see docs/FAILURE_ANALYSIS.md's mechanism section. The 3 llama_*_1.5b_user
transcripts are from that earlier, pre-matched-simulator (confounded)
run - restored from archive into data/raw/ and data/runs/, kept distinct
from the matched-simulator llama_task{0,1,3}.json files used elsewhere.
Reuses helpers from failure_analysis.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import failure_analysis as fa

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCES = {
    "qwen_task0": DATA_DIR / "qwen_task0.json",
    "qwen_task1": DATA_DIR / "qwen_task1.json",
    "qwen_task2": DATA_DIR / "qwen_task2.json",
    "qwen_task3": DATA_DIR / "qwen_task3.json",
    # archived pre-matched-simulator run (Qwen2.5-1.5B-Instruct fp16 user).
    "llama_task0": DATA_DIR / "llama_task0_1.5b_user.json",
    "llama_task1": DATA_DIR / "llama_task1_1.5b_user.json",
    "llama_task3": DATA_DIR / "llama_task3_1.5b_user.json",
}

USER_MODEL_BY_CONVO = {
    "qwen_task0": "Qwen2.5-7B-AWQ", "qwen_task1": "Qwen2.5-7B-AWQ",
    "qwen_task2": "Qwen2.5-7B-AWQ", "qwen_task3": "Qwen2.5-7B-AWQ",
    "llama_task0": "Qwen2.5-1.5B-fp16", "llama_task1": "Qwen2.5-1.5B-fp16",
    "llama_task3": "Qwen2.5-1.5B-fp16",
}


def analyze(name: str, path: str) -> dict:
    messages, _policy = fa.load_messages(path)
    user_texts = [m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]

    # every distinct ID-shaped value that EVER appears as a tool-call
    # argument in this conversation (grounded or not) - the candidate
    # vocabulary. Of those, how many appear literally in a user
    # utterance ANYWHERE in the conversation (not causally restricted -
    # this is a mechanism check, not the grounding metric itself).
    all_id_values: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                for _field, value in fa.id_values(tc.get("arguments") or {}):
                    all_id_values.add(value)

    stated_by_user = {v for v in all_id_values if any(v in t for t in user_texts)}

    return {
        "name": name,
        "user_model": USER_MODEL_BY_CONVO[name],
        "n_user_turns": len(user_texts),
        "distinct_id_values_in_tool_calls": len(all_id_values),
        "distinct_id_values_stated_by_user": len(stated_by_user),
        "fraction_of_id_vocabulary_stated_by_user": (
            len(stated_by_user) / len(all_id_values) if all_id_values else None
        ),
    }


def main() -> None:
    results = {}
    by_user_model: dict[str, list[int]] = {}
    for name, path in SOURCES.items():
        if not Path(path).exists():
            print(f"MISSING: {name} -> {path}")
            continue
        r = analyze(name, path)
        results[name] = r
        by_user_model.setdefault(r["user_model"], []).append(r["distinct_id_values_stated_by_user"])
        frac = r["fraction_of_id_vocabulary_stated_by_user"]
        print(f"{name} (user={r['user_model']}, {r['n_user_turns']} user turns): "
              f"{r['distinct_id_values_stated_by_user']}/{r['distinct_id_values_in_tool_calls']} distinct ID values "
              f"also stated by the user ({frac:.1%})" if frac is not None else
              f"{name}: no ID values in tool calls")

    print("\n=== by user model ===")
    for user_model, counts in by_user_model.items():
        print(f"{user_model}: per-conversation user-stated-ID counts = {counts}, "
              f"mean = {sum(counts)/len(counts):.2f}, total = {sum(counts)}")

    Path("user_id_mentions_raw.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nwrote user_id_mentions_raw.json")


if __name__ == "__main__":
    main()
