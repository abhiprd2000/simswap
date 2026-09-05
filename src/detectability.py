"""Sweep planted interaction delta x grid size on fake data.

Finds the smallest interaction magnitude the omnibus LRT + targeted
contrast tests can detect, at a few candidate grid sizes. This is the
design justification for the grid size and minimum detectable delta in
ANALYSIS_PLAN.md - deterministic and rerunnable, not scratch.
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze
import fake_runs

WORK_DIR = Path("_detectability_work")
TABLES_DIR = Path("tables")

GRIDS = [(15, 2), (20, 2), (20, 3)]
DELTAS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]


def is_primary(row: dict, agent_a: str, agent_b: str, flip_user: str) -> bool:
    pair = {row["agent_i"], row["agent_j"]}
    users = {row["user_a"], row["user_b"]}
    return pair == {agent_a, agent_b} and flip_user in users


def touches_pair_and_flip(row: dict, agent_a: str, agent_b: str, flip_user: str) -> bool:
    pair = {row["agent_i"], row["agent_j"]}
    users = {row["user_a"], row["user_b"]}
    return bool(pair & {agent_a, agent_b}) and flip_user in users


def run_one(n_tasks: int, n_seeds: int, delta: float) -> dict:
    tag = f"t{n_tasks}_s{n_seeds}_d{delta}"
    runs_dir = WORK_DIR / f"runs_{tag}"
    tables_dir = WORK_DIR / f"tables_{tag}"
    figures_dir = WORK_DIR / f"figures_{tag}"
    for d in (runs_dir, tables_dir, figures_dir):
        if d.exists():
            shutil.rmtree(d)

    argv_bak = sys.argv
    sys.argv = ["fake_runs.py", "--n-tasks", str(n_tasks), "--n-seeds", str(n_seeds),
                "--delta", str(delta), "--out", str(runs_dir)]
    try:
        fake_runs.main()
    finally:
        sys.argv = argv_bak

    ground_truth = json.loads((runs_dir / "ground_truth.json").read_text())
    agent_a, agent_b = ground_truth["inversion_agents"]
    flip_user = ground_truth["flipped_under_user_model"]

    argv_bak = sys.argv
    sys.argv = ["analyze.py", "--runs-dir", str(runs_dir), "--tables-dir", str(tables_dir),
                "--figures-dir", str(figures_dir), "--n-perm", "2000"]
    try:
        analyze.main()
    except SystemExit:
        pass  # analyze.py exits 1 if any output failed; tables written so far are still read below
    finally:
        sys.argv = argv_bak

    with open(tables_dir / "omnibus_lrt.csv") as f:
        lrt_row = next(csv.DictReader(f))
    lrt_p = float(lrt_row["p_value"]) if lrt_row["p_value"] not in ("", "nan") else None

    with open(tables_dir / "targeted_contrasts.csv") as f:
        contrasts = list(csv.DictReader(f))

    primary_rows = [c for c in contrasts if is_primary(c, agent_a, agent_b, flip_user)]
    wrong_rows = [c for c in contrasts if not touches_pair_and_flip(c, agent_a, agent_b, flip_user)]

    def holm_sig(c: dict) -> bool:
        return c["holm_p"] not in ("", "nan") and float(c["holm_p"]) < 0.05

    # matches ANALYSIS_PLAN.md's decision rule exactly: LRT significant AND
    # at least one targeted contrast survives Holm AND it's on the planted
    # pair. An unrelated contrast also surviving Holm is expected background
    # noise (documented in ANALYSIS_PLAN.md), not a disqualifier - tracked
    # here for information, not part of the detected/not-detected gate.
    primary_detected = bool(primary_rows) and any(holm_sig(c) for c in primary_rows)
    false_positive = any(holm_sig(c) for c in wrong_rows)
    lrt_sig = lrt_p is not None and lrt_p < 0.05
    detected = lrt_sig and primary_detected

    shutil.rmtree(runs_dir)
    shutil.rmtree(tables_dir)
    shutil.rmtree(figures_dir)

    return {
        "n_tasks": n_tasks, "n_seeds": n_seeds, "delta": delta,
        "lrt_p": lrt_p, "n_primary_contrasts": len(primary_rows),
        "primary_detected": primary_detected, "false_positive": false_positive,
        "detected": detected,
    }


def main() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    results = []
    for (n_tasks, n_seeds), delta in itertools.product(GRIDS, DELTAS):
        r = run_one(n_tasks, n_seeds, delta)
        results.append(r)
        print(f"t{n_tasks} s{n_seeds} delta={delta}: lrt_p={r['lrt_p']} "
              f"primary_detected={r['primary_detected']} false_positive={r['false_positive']} "
              f"-> detected={r['detected']}")

    shutil.rmtree(WORK_DIR)

    with open(TABLES_DIR / "detectability.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== smallest detected delta per grid ===")
    summary = []
    for n_tasks, n_seeds in GRIDS:
        detected = [r["delta"] for r in results if r["n_tasks"] == n_tasks and r["n_seeds"] == n_seeds and r["detected"]]
        smallest = min(detected) if detected else None
        summary.append({"n_tasks": n_tasks, "n_seeds": n_seeds, "smallest_detected_delta": smallest})
        print(f"{n_tasks}x{n_seeds}: {smallest if smallest is not None else 'NONE DETECTED'}")

    with open(TABLES_DIR / "detectability_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_tasks", "n_seeds", "smallest_detected_delta"])
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
