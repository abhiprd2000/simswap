"""Exact paired permutation (sign-flip) test on the grounding rate gap,
tasks {0,1,3}, both models facing the same user (Qwen2.5-7B-Instruct-AWQ).
Statistic per task = qwen_flat_rate - llama_flat_rate (flat, not
conditional: qwen_task0's conditional rate is undefined - see
docs/FAILURE_ANALYSIS.md). p = (b+1)/(n+1) over all 2**3=8 exact sign
assignments. Reads failure_analysis_raw.json; run failure_analysis.py first.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

TASKS = ["task0", "task1", "task3"]


def main() -> None:
    raw = json.loads(Path("failure_analysis_raw.json").read_text())

    diffs = []
    for t in TASKS:
        q = raw[f"qwen_{t}"]["ungrounded_fraction_by_argument"]
        l = raw[f"llama_{t}"]["ungrounded_fraction_by_argument"]
        d = q - l
        diffs.append(d)
        print(f"{t}: qwen_flat={q:.4f} llama_flat={l:.4f} diff(qwen-llama)={d:+.4f}")

    observed = sum(diffs)
    print(f"\nobserved sum of signed diffs = {observed:+.4f} (all {len(diffs)}/{len(diffs)} tasks same direction: "
          f"{'qwen higher' if all(d > 0 for d in diffs) else 'llama higher' if all(d < 0 for d in diffs) else 'mixed'})")

    abs_diffs = [abs(d) for d in diffs]
    n_perms = 0
    b_two_sided = 0
    b_one_sided = 0  # sum >= observed (observed is positive: qwen consistently higher)
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        stat = sum(s * a for s, a in zip(signs, abs_diffs))
        n_perms += 1
        if abs(stat) >= abs(observed) - 1e-12:
            b_two_sided += 1
        if stat >= observed - 1e-12:
            b_one_sided += 1

    # exclude the observed assignment itself from b before the +1 correction
    b_two_sided -= 1
    b_one_sided -= 1

    p_two_sided = (b_two_sided + 1) / (n_perms + 1)
    p_one_sided = (b_one_sided + 1) / (n_perms + 1)

    print(f"\nexact enumeration: n={n_perms} sign assignments")
    print(f"two-sided: b={b_two_sided}, p=(b+1)/(n+1)={p_two_sided:.4f}")
    print(f"one-sided (H1: qwen rate > llama rate, matching the direction observed "
          f"in all 3 tasks): b={b_one_sided}, p=(b+1)/(n+1)={p_one_sided:.4f}")
    print(f"\nfloor of this test at n=3 paired tasks: the smallest possible p-value "
          f"is 1/{n_perms + 1} = {1/(n_perms+1):.4f} (one-sided) — even a perfectly "
          f"consistent 3/3 result cannot reach p<0.05 with this few pairs.")

    Path("permutation_test_result.json").write_text(json.dumps({
        "tasks": TASKS, "diffs_qwen_minus_llama": diffs, "observed_sum": observed,
        "n_permutations": n_perms, "b_two_sided": b_two_sided, "b_one_sided": b_one_sided,
        "p_two_sided": p_two_sided, "p_one_sided": p_one_sided,
    }, indent=2))
    print("\nwrote permutation_test_result.json")


if __name__ == "__main__":
    main()
