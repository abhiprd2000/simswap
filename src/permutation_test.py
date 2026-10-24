"""Exact paired permutation (sign-flip) test on the grounding rate gap,
tasks {0,1,3}, both models facing the same user (Qwen2.5-7B-Instruct-AWQ).
Statistic per task = qwen_flat_rate - llama_flat_rate (flat, not
conditional: qwen_task0's conditional rate is undefined - see
docs/FAILURE_ANALYSIS.md).

p = count(assignments with stat >= observed, INCLUDING the observed
assignment itself) / n, over all 2**3=8 exact sign assignments. This is
EXHAUSTIVE enumeration, not a Monte Carlo sample - every possible sign
assignment is examined, so the observed assignment is already counted
as one of the n outcomes and belongs in the numerator like any other.
The "(b+1)/(n+1)" add-one correction (fixed 2026-10-24, was used here
before) is for approximate permutation tests built from a *finite random
sample* of assignments, to guarantee a nonzero p-value when the sample
might miss the true tail; it does not apply to a full enumeration,
where n already reflects every possible assignment. Reads
failure_analysis_raw.json; run failure_analysis.py first.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

TASKS = ["task0", "task1", "task3"]


def exact_perm_test(diffs: list[float]) -> dict:
    """Exhaustive sign-flip enumeration over all 2**len(diffs) assignments.
    p = b/n, b including the observed assignment itself - this is a full
    enumeration, not a Monte Carlo sample, so no add-one correction (see
    module docstring). Shared by permutation_test.py and paper_numbers.py
    so the formula can't drift between the two call sites again."""
    observed = sum(diffs)
    abs_diffs = [abs(d) for d in diffs]
    n_perms = 0
    b_two_sided = 0
    b_one_sided = 0  # stat >= observed (observed is positive here: qwen consistently higher)
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        stat = sum(s * a for s, a in zip(signs, abs_diffs))
        n_perms += 1
        if abs(stat) >= abs(observed) - 1e-12:
            b_two_sided += 1
        if stat >= observed - 1e-12:
            b_one_sided += 1
    return {
        "observed": observed, "n_perms": n_perms,
        "b_two_sided": b_two_sided, "b_one_sided": b_one_sided,
        "p_two_sided": b_two_sided / n_perms, "p_one_sided": b_one_sided / n_perms,
        "all_same_sign": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
    }


def main() -> None:
    raw = json.loads(Path("failure_analysis_raw.json").read_text())

    diffs = []
    for t in TASKS:
        q = raw[f"qwen_{t}"]["ungrounded_fraction_by_argument"]
        l = raw[f"llama_{t}"]["ungrounded_fraction_by_argument"]
        d = q - l
        diffs.append(d)
        print(f"{t}: qwen_flat={q:.4f} llama_flat={l:.4f} diff(qwen-llama)={d:+.4f}")

    r = exact_perm_test(diffs)
    print(f"\nobserved sum of signed diffs = {r['observed']:+.4f} (all {len(diffs)}/{len(diffs)} tasks same direction: "
          f"{'qwen higher' if all(d > 0 for d in diffs) else 'llama higher' if all(d < 0 for d in diffs) else 'mixed'})")

    print(f"\nexact enumeration: n={r['n_perms']} sign assignments")
    print(f"two-sided: b={r['b_two_sided']}, p=b/n={r['p_two_sided']:.4f}")
    print(f"one-sided (H1: qwen rate > llama rate, matching the direction observed "
          f"in all 3 tasks): b={r['b_one_sided']}, p=b/n={r['p_one_sided']:.4f}")
    print(f"\nfloor of this test at n=3 paired tasks: the smallest possible p-value "
          f"is 1/{r['n_perms']} = {1/r['n_perms']:.4f} (one-sided) — even a perfectly "
          f"consistent 3/3 result cannot reach p<0.05 with this few pairs.")

    Path("permutation_test_result.json").write_text(json.dumps({
        "tasks": TASKS, "diffs_qwen_minus_llama": diffs, "observed_sum": r["observed"],
        "n_permutations": r["n_perms"], "b_two_sided": r["b_two_sided"], "b_one_sided": r["b_one_sided"],
        "p_two_sided": r["p_two_sided"], "p_one_sided": r["p_one_sided"],
    }, indent=2))
    print("\nwrote permutation_test_result.json")


if __name__ == "__main__":
    main()
