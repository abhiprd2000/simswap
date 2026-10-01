"""Argument-type schema conformance: does a tool call's JSON types match the
tool's declared schema, independent of whether the IDs in it are grounded?
The retail domain's only List[str] arguments are item_ids/new_item_ids;
every other argument is a plain str. See docs/SCHEMA_CONFORMANCE.md.
"""

from __future__ import annotations

import json
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
}

# vLLM's llama3_json parser hard-rejects any request where the model attempts
# multiple simultaneous tool_calls in one turn ("This model only supports
# single tool-calls at once!"). No transcript exists for this - the rejection
# happens before tau2 records anything - reproduced on 2 independent attempts
# against an earlier (superseded) user simulator; not retried against the
# matched user, since this is a parser-level limitation independent of the
# user's content. See docs/FAILURE_ANALYSIS.md's "CONFOUND CHECK" section.
NON_TRANSCRIPT_EVENTS = [
    {
        "conversation": "llama_task2", "model": "llama", "attempt": 1,
        "malformation": "parallel_tool_calls_rejected",
        "detail": "vLLM llama3_json parser 400: \"This model only supports single tool-calls at once!\"",
        "source": "archived pre-matched-simulator run, not included in this repository",
    },
    {
        "conversation": "llama_task2", "model": "llama", "attempt": 2,
        "malformation": "parallel_tool_calls_rejected",
        "detail": "vLLM llama3_json parser 400: \"This model only supports single tool-calls at once!\"",
        "source": "archived pre-matched-simulator run, not included in this repository",
    },
]

LIST_FIELDS = {"item_ids", "new_item_ids"}
LIST_BEARING_TOOLS = {"exchange_delivered_order_items", "modify_pending_order_items", "return_delivered_order_items"}


def classify_arg(field: str, value) -> str | None:
    """Return a malformation label, or None if the arg's JSON type matches schema."""
    expects_list = field in LIST_FIELDS
    is_list = isinstance(value, list)
    is_str = isinstance(value, str)
    if expects_list and not is_list:
        return f"{field}: str-instead-of-list" if is_str else f"{field}: {type(value).__name__}-instead-of-list"
    if not expects_list and is_list:
        return f"{field}: list-instead-of-str"
    if expects_list and is_list:
        bad_items = [type(x).__name__ for x in value if not isinstance(x, str)]
        if bad_items:
            return f"{field}: list-contains-non-str-items({bad_items})"
    return None


def analyze(path) -> tuple[list[dict], int]:
    messages = json.loads(Path(path).read_text())["sim_dump"]["messages"]
    findings = []
    n_list_bearing_calls = 0
    pending_calls = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args = tc.get("arguments") or {}
                if tc["name"] in LIST_BEARING_TOOLS:
                    n_list_bearing_calls += 1
                malformations = [lbl for k, v in args.items() if (lbl := classify_arg(k, v))]
                pending_calls.append({"name": tc["name"], "arguments": args, "malformations": malformations})
        elif m.get("role") == "tool":
            if pending_calls:
                call = pending_calls.pop(0)
                if call["malformations"]:
                    findings.append({
                        "tool": call["name"], "arguments": call["arguments"],
                        "malformations": call["malformations"],
                        "result_was_error": bool(m.get("error")),
                        "result_content": m.get("content"),
                    })
    return findings, n_list_bearing_calls


def main() -> None:
    per_model = {"qwen": {"convos": 0, "malformed_calls": 0, "list_bearing_calls": 0, "types": {}},
                 "llama": {"convos": 0, "malformed_calls": 0, "list_bearing_calls": 0, "types": {}}}

    for name, path in SOURCES.items():
        model = "qwen" if name.startswith("qwen") else "llama"
        per_model[model]["convos"] += 1
        findings, n_list_bearing = analyze(path)
        per_model[model]["list_bearing_calls"] += n_list_bearing
        if findings:
            print(f"\n=== {name}: {len(findings)} malformed call(s) ===")
            for f in findings:
                print(f"  tool={f['tool']} malformations={f['malformations']} "
                      f"result_was_error={f['result_was_error']} result={str(f['result_content'])[:100]!r}")
                per_model[model]["malformed_calls"] += 1
                for lbl in f["malformations"]:
                    kind = lbl.split(":")[1].strip().split("(")[0]
                    per_model[model]["types"][kind] = per_model[model]["types"].get(kind, 0) + 1
        else:
            print(f"\n=== {name}: 0 malformed calls ===")

    print("\n\n=== SUMMARY: argument-type malformation ===")
    for model, stats in per_model.items():
        rate = (stats['malformed_calls'] / stats['list_bearing_calls']) if stats['list_bearing_calls'] else None
        print(f"{model}: {stats['malformed_calls']}/{stats['list_bearing_calls']} malformed "
              f"({rate}) across {stats['convos']} conversation(s), types={stats['types']}")

    print("\n=== SUMMARY: request-structure rejection (no transcript) ===")
    by_model: dict[str, list[dict]] = {"qwen": [], "llama": []}
    for e in NON_TRANSCRIPT_EVENTS:
        by_model[e["model"]].append(e)
    for model, events in by_model.items():
        if not events:
            print(f"{model}: 0 request-structure rejections observed")
            continue
        for e in events:
            print(f"{model}: {e['conversation']} attempt {e['attempt']} - {e['malformation']}: {e['detail']}")

    Path("schema_conformance_raw.json").write_text(json.dumps(
        {"argument_type_malformation": per_model, "request_structure_rejections": NON_TRANSCRIPT_EVENTS},
        indent=2, default=str,
    ))
    print("\nwrote schema_conformance_raw.json")


if __name__ == "__main__":
    main()
