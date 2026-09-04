"""Generate synthetic runs/ matching schema.py, for testing analyze.py.

Default mode plants a known rank inversion between two agents under one
user model, and prints the ground truth so analyze.py's recovery of it
can be checked. --null mode has no inversion, only noise - analyze.py
must not report one there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from schema import ActionTally, RunRecord

FAMILIES = ["qwen", "llama", "mistral", "gemma", "phi", "yi", "deepseek", "olmo"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--n-users", type=int, default=4)
    p.add_argument("--n-tasks", type=int, default=15)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--null", action="store_true", help="no planted inversion, noise only")
    p.add_argument("--delta", type=float, default=0.30,
                    help="interaction magnitude for the planted inversion (p-scale, applied as +-delta)")
    p.add_argument("--rng-seed", type=int, default=42, help="seed for the fake data generator itself")
    p.add_argument("--domain", default="retail")
    p.add_argument("--tau2-version", default="1.0.1")
    p.add_argument("--tau2-commit", default="fake0000tau2")
    p.add_argument("--simswap-commit", default="fake0000simswap")
    p.add_argument("--gpu-name", default="fake0000gpu")
    return p.parse_args()


def make_models(n: int, prefix: str) -> list[dict]:
    """Family = last token of the name, e.g. 'agent0-qwen' -> 'qwen'."""
    out = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        name = f"{prefix}{i}-{family}"
        revision = f"rev-{abs(hash(name)) % 10**8:08x}"
        out.append({"name": name, "family": family, "revision": revision})
    return out


def agent_skills(n: int, null_mode: bool) -> list[float]:
    """Base action-correctness skill per agent, band roughly 20-60%.

    Gaps between agents must clear the per-cell noise floor or a rank
    "flip" is just noise. With ~13-15 (task, seed) samples per cell and
    small per-run action counts (~3-13), a single run's action_accuracy
    has binomial sampling std ~0.15-0.2, so a cell mean over ~27 runs
    still has std ~0.03-0.04. Adjacent-agent gaps below ~0.10 are not
    reliably stable against that.
    """
    if n >= 2:
        first_two = [0.55, 0.38] if null_mode else [0.50, 0.40]
    else:
        first_two = [0.55] if null_mode else [0.50]
    rest_n = max(n - 2, 0)
    if rest_n > 0:
        # background agents (not part of the tested pair) are spaced far
        # apart on purpose - see module docstring math. a ~0.15 gap still
        # flips occasionally under a ~27-sample permuted bucket; ~0.25+
        # does not.
        hi, lo = (0.10, -0.20) if null_mode else (0.16, -0.02)
        rest = list(np.linspace(hi, lo, rest_n))
    else:
        rest = []
    skills = (first_two[:n]) + rest
    return skills[:n]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.rng_seed)

    agents = make_models(args.n_agents, "agent")
    users = make_models(args.n_users, "user")
    tasks = [f"task{i:02d}" for i in range(args.n_tasks)]
    seeds = list(range(args.n_seeds))

    skills = agent_skills(args.n_agents, args.null)
    user_leniency = list(np.linspace(0.05, -0.05, args.n_users)) if args.n_users > 1 else [0.0]

    # task difficulty offsets; a slice of tasks near-floor for everyone
    task_difficulty = rng.normal(0, 0.05, args.n_tasks)
    n_floor = max(1, round(args.n_tasks * 0.13)) if args.n_tasks > 0 else 0
    floor_idx = rng.choice(args.n_tasks, size=n_floor, replace=False) if n_floor else []
    for i in floor_idx:
        task_difficulty[i] = -0.35

    # planted inversion: agents[0] vs agents[1] flip order under the last
    # user model only. all other user models keep agents[0] > agents[1].
    inversion_user_idx = args.n_users - 1
    ground_truth = {"mode": "null" if args.null else "default"}
    if not args.null and args.n_agents >= 2 and args.n_users >= 2:
        ground_truth.update(
            {
                "inversion_agents": [agents[0]["name"], agents[1]["name"]],
                "flipped_under_user_model": users[inversion_user_idx]["name"],
                "consistent_under_user_models": [
                    u["name"] for i, u in enumerate(users) if i != inversion_user_idx
                ],
                "interaction_delta": args.delta,
                "note": (
                    f"{agents[0]['name']} > {agents[1]['name']} everywhere except "
                    f"under {users[inversion_user_idx]['name']}, where it flips "
                    f"(interaction delta = +-{args.delta})"
                ),
            }
        )

    talk_base = np.linspace(6, 14, args.n_users) if args.n_users > 1 else np.array([8.0])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transcripts").mkdir(parents=True, exist_ok=True)

    status_counts = {"completed": 0, "error": 0, "timeout": 0}
    reason_counts: dict[str, int] = {}
    n_written = 0

    for ai, agent in enumerate(agents):
        for ui, user in enumerate(users):
            interaction = 0.0
            if not args.null and ai == 0 and ui == inversion_user_idx:
                interaction = -args.delta
            elif not args.null and ai == 1 and ui == inversion_user_idx:
                interaction = args.delta

            for ti, task_id in enumerate(tasks):
                for seed in seeds:
                    run_id = f"{task_id}__{agent['name']}__{user['name']}__seed{seed}"

                    p = skills[ai] + user_leniency[ui] + interaction + task_difficulty[ti]
                    p += rng.normal(0, 0.03)
                    p = float(np.clip(p, 0.02, 0.95))

                    n_turns = max(2, int(round(talk_base[ui] + rng.integers(-2, 3))))

                    dice = rng.random()
                    if dice < 0.05:
                        status, reason = "error", "harness_crash"
                        error = "Traceback (most recent call last):\n  RuntimeError: vLLM server connection reset"
                        reward = None
                        task_success = None
                        read = ActionTally(0, 0)
                        write = ActionTally(0, 0)
                        n_turns = min(n_turns, int(rng.integers(0, 3)))
                        wall_clock = round(float(rng.uniform(1, 15)), 2)
                    elif dice < 0.08:
                        status, reason = "completed", "max_turns"
                        error = None
                        reward = 0.0
                        task_success = False
                        n_read = int(rng.integers(1, 5))
                        n_write = int(rng.integers(1, 4))
                        read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                        write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                        n_turns = max(n_turns, 20 + int(rng.integers(0, 5)))
                        wall_clock = round(float(rng.uniform(30, 90)), 2)
                    elif dice < 0.10:
                        status, reason = "error", "agent_error"
                        error = "Traceback (most recent call last):\n  ValueError: malformed tool call arguments"
                        reward = None
                        task_success = None
                        n_read = int(rng.integers(0, 3))
                        n_write = int(rng.integers(0, 2))
                        read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                        write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                        wall_clock = round(float(rng.uniform(2, 20)), 2)
                    else:
                        status, reason = "completed", "normal"
                        error = None
                        p_success = p**1.5
                        reward = 1.0 if rng.random() < p_success else 0.0
                        task_success = reward >= 1.0 - 1e-6
                        n_read = int(rng.integers(4, 14))
                        n_write = int(rng.integers(3, 10))
                        read = ActionTally(n_read, int(rng.binomial(n_read, p)))
                        write = ActionTally(n_write, int(rng.binomial(n_write, p)))
                        wall_clock = round(float(rng.uniform(15, 60)), 2)

                    status_counts[status] += 1
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

                    transcript_rel = f"transcripts/{run_id}.json"
                    transcript = [
                        {"turn": t, "role": "user" if t % 2 == 0 else "assistant", "text": "..."}
                        for t in range(n_turns)
                    ]
                    (args.out / transcript_rel).write_text(json.dumps(transcript))

                    record = RunRecord(
                        schema_version=1,
                        run_id=run_id,
                        tau2_version=args.tau2_version,
                        tau2_commit=args.tau2_commit,
                        simswap_commit=args.simswap_commit,
                        task_id=task_id,
                        domain=args.domain,
                        agent_model=agent["name"],
                        agent_model_revision=agent["revision"],
                        user_model=user["name"],
                        user_model_revision=user["revision"],
                        agent_quantization="fake-quant",
                        user_quantization="fake-quant",
                        seed=seed,
                        temperature=0.0,
                        gpu_name=args.gpu_name,
                        timestamp="2026-08-14T00:00:00Z",
                        max_steps=200,
                        status=status,
                        termination_reason=reason,
                        error=error,
                        task_success=task_success,
                        reward=reward,
                        reward_info_note=None,
                        n_turns=n_turns,
                        read=read,
                        write=write,
                        wall_clock_seconds=wall_clock,
                        agent_cost=0.0,
                        user_cost=0.0,
                        transcript_path=transcript_rel,
                    )
                    record.save(args.out / f"{run_id}.json")
                    n_written += 1

    (args.out / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    print(f"wrote {n_written} runs to {args.out}")
    print(f"status counts: {status_counts}")
    print(f"termination reason counts: {reason_counts}")
    print(f"agents: {[a['name'] for a in agents]}")
    print(f"users: {[u['name'] for u in users]}")
    print(f"agent skills (base): {dict(zip((a['name'] for a in agents), skills))}")
    print("GROUND TRUTH:")
    print(json.dumps(ground_truth, indent=2))


if __name__ == "__main__":
    main()
