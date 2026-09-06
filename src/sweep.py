"""Run one (agent, user model) pair through tau2-bench retail, 20 tasks x 2 seeds.

Runs ONE pair per invocation against vLLM servers you already started (one per GPU) -
starting/cycling servers is an outer loop, not this script's job. Resumable: re-running
skips any (task, seed) cell that already has a run file on disk, checkpointed per episode.
See README.md's "Reproducing the raw GPU runs" section for the exact setup and command.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from schema import ActionTally, RunRecord

TAU2_VERSION_PINNED = "1.0.1"
TAU2_COMMIT_PINNED = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

# tau2.config.DEFAULT_MAX_STEPS at the pinned commit is 200 - set explicitly
# rather than relying on tau2's default, and record it on every run so the
# cap is visible (Gate 1 Pass A hit it: 201 turns = 200 steps + greeting).
MAX_STEPS = 200

GPU_NAME_DRY_RUN = "DRY_RUN_NO_GPU"

DOMAIN = "retail"
N_TASKS = 20
SEEDS = (0, 1)
TASK_SAMPLE_SEED = 20260814  # date this grid was pre-registered, ANALYSIS_PLAN.md

# tau2.data_model.simulation.TerminationReason -> our normalized set.
# Keyed by the enum's .value (lowercase snake_case), NOT its .name -
# verified against source at the pinned commit at Gate 1, 2026-08-15.
# normal: a clean, expected end of the conversation (either side signaled
# done). max_turns: hit the step cap. agent_error: the agent broke
# something. harness_crash: infra/timeout/unexpected - not a clean tau2
# signal about agent or user behavior.
TERMINATION_REASON_MAP = {
    "user_stop": "normal",
    "agent_stop": "normal",
    "max_steps": "max_turns",
    "too_many_errors": "agent_error",
    "agent_error": "agent_error",
    "user_error": "harness_crash",
    "timeout": "harness_crash",
    "infrastructure_error": "harness_crash",
    "context_window_exceeded": "harness_crash",
    "unexpected_error": "harness_crash",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-model", required=False)
    p.add_argument("--agent-revision", required=False)
    p.add_argument("--agent-url", required=False, help="vLLM OpenAI-compatible base_url, e.g. http://localhost:8001/v1")
    p.add_argument("--agent-quantization", required=False, help='e.g. "fp16", "awq-int4"')
    p.add_argument("--user-model", required=False)
    p.add_argument("--user-revision", required=False)
    p.add_argument("--user-url", required=False)
    p.add_argument("--user-quantization", required=False, help='e.g. "fp16", "awq-int4"')
    p.add_argument("--domain", default=DOMAIN)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    p.add_argument("--llm-timeout-seconds", type=float, default=120.0)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--print-tasks", action="store_true", help="print the selected task ids and exit")
    p.add_argument("--dry-run", action="store_true",
                    help="mock the LLM call, no GPU, exercise the real write path, one fake record to --out")
    args = p.parse_args()
    if args.print_tasks or args.dry_run:
        return args
    missing = [n for n in ("agent_model", "agent_revision", "agent_url", "agent_quantization",
                            "user_model", "user_revision", "user_url", "user_quantization")
               if getattr(args, n) is None]
    if missing:
        p.error(f"missing required args: {missing}")
    return args


def check_tau2_version() -> None:
    try:
        installed = importlib.metadata.version("tau2")
    except importlib.metadata.PackageNotFoundError:
        installed = importlib.metadata.version("tau2-bench")
    if installed != TAU2_VERSION_PINNED:
        raise RuntimeError(
            f"tau2 version mismatch: pinned {TAU2_VERSION_PINNED}, installed {installed}. "
            "Never upgrade mid-sweep - see ANALYSIS_PLAN.md."
        )


def get_simswap_commit() -> str:
    repo_dir = Path(__file__).parent
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()


def get_gpu_name() -> str:
    """One GPU model for the box, via nvidia-smi. T4 vs P100 fp16 differ at temp=0."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    ).decode()
    names = {line.strip() for line in out.splitlines() if line.strip()}
    if not names:
        raise RuntimeError("nvidia-smi reported no GPUs")
    if len(names) > 1:
        raise RuntimeError(f"mixed GPU models on this box: {sorted(names)} - sweep needs one model")
    return names.pop()


def check_gpu_consistency(out_dir: Path, gpu_name: str) -> None:
    """Every run file already on disk must match this session's GPU model."""
    for p in out_dir.glob("*.json"):
        if p.name == "ground_truth.json":
            continue
        record = RunRecord.load(p)
        if record.gpu_name == GPU_NAME_DRY_RUN:
            continue
        if record.gpu_name != gpu_name:
            raise RuntimeError(
                f"GPU mismatch: {p.name} was run on {record.gpu_name!r}, "
                f"this session has {gpu_name!r}. Do not mix GPU models in one sweep."
            )


def select_tasks(domain: str, n: int) -> list[str]:
    from tau2.runner import get_tasks

    tasks = get_tasks(domain, task_split_name="base")
    ids = sorted((t.id for t in tasks), key=lambda x: int(x) if x.isdigit() else x)
    return random.Random(TASK_SAMPLE_SEED).sample(ids, n)


def build_config(args: argparse.Namespace, seed: int):
    from tau2.data_model.simulation import TextRunConfig

    llm_args = {"temperature": 0.0, "timeout": args.llm_timeout_seconds}
    return TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=f"openai/{args.agent_model}",
        llm_args_agent={**llm_args, "api_base": args.agent_url, "api_key": "EMPTY"},
        llm_user=f"openai/{args.user_model}",
        llm_args_user={**llm_args, "api_base": args.user_url, "api_key": "EMPTY"},
        seed=seed,
        max_steps=args.max_steps,
    )


def action_tally(partial: dict | None, key: str) -> ActionTally:
    if partial is None:
        return ActionTally(0, 0)
    sub = partial.get(key)
    if sub is None:
        return ActionTally(0, 0)
    return ActionTally(count=sub["count"], correct=sub["correct"])


def convert_success(sim, run_id: str, task_id: str, args: argparse.Namespace, seed: int,
                     simswap_commit: str, gpu_name: str, transcript_path: str) -> RunRecord:
    reward_info = sim.reward_info
    if reward_info is None:
        raise RuntimeError(f"run {run_id}: completed simulation has no reward_info")
    partial = reward_info.partial_action_reward
    reason = TERMINATION_REASON_MAP[sim.termination_reason.value if hasattr(sim.termination_reason, "value") else sim.termination_reason]

    return RunRecord(
        schema_version=1,
        run_id=run_id,
        tau2_version=TAU2_VERSION_PINNED,
        tau2_commit=TAU2_COMMIT_PINNED,
        simswap_commit=simswap_commit,
        task_id=task_id,
        domain=args.domain,
        agent_model=args.agent_model,
        agent_model_revision=args.agent_revision,
        user_model=args.user_model,
        user_model_revision=args.user_revision,
        agent_quantization=args.agent_quantization,
        user_quantization=args.user_quantization,
        seed=seed,
        temperature=0.0,
        gpu_name=gpu_name,
        timestamp=str(sim.timestamp),
        max_steps=args.max_steps,
        status="completed",
        termination_reason=reason,
        error=None,
        task_success=bool(reward_info.reward >= 1.0 - 1e-6),
        reward=float(reward_info.reward),
        reward_info_note=getattr(reward_info, "info", None),
        n_turns=len(sim.get_messages()),
        read=action_tally(partial, "read"),
        write=action_tally(partial, "write"),
        wall_clock_seconds=float(sim.duration) if sim.duration is not None else None,
        agent_cost=float(sim.agent_cost) if sim.agent_cost is not None else None,
        user_cost=float(sim.user_cost) if sim.user_cost is not None else None,
        transcript_path=transcript_path,
    )


def convert_failure(run_id: str, task_id: str, args: argparse.Namespace, seed: int,
                     simswap_commit: str, gpu_name: str, transcript_path: str, exc: Exception) -> RunRecord:
    is_timeout = "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower()
    return RunRecord(
        schema_version=1,
        run_id=run_id,
        tau2_version=TAU2_VERSION_PINNED,
        tau2_commit=TAU2_COMMIT_PINNED,
        simswap_commit=simswap_commit,
        task_id=task_id,
        domain=args.domain,
        agent_model=args.agent_model,
        agent_model_revision=args.agent_revision,
        user_model=args.user_model,
        user_model_revision=args.user_revision,
        agent_quantization=args.agent_quantization,
        user_quantization=args.user_quantization,
        seed=seed,
        temperature=0.0,
        gpu_name=gpu_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        max_steps=args.max_steps,
        status="timeout" if is_timeout else "error",
        termination_reason="harness_crash",
        error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        task_success=None,
        reward=None,
        reward_info_note=None,
        n_turns=0,
        read=ActionTally(0, 0),
        write=ActionTally(0, 0),
        wall_clock_seconds=None,
        agent_cost=None,
        user_cost=None,
        transcript_path=transcript_path,
    )


def run_cell(args: argparse.Namespace, task, seed: int, run_id: str, simswap_commit: str,
             gpu_name: str, transcripts_dir: Path) -> RunRecord:
    from tau2.runner import run_single_task

    transcript_rel = f"transcripts/{run_id}.json"
    last_exc: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            config = build_config(args, seed)
            sim = run_single_task(config, task, seed=seed)
            (transcripts_dir / f"{run_id}.json").write_text(
                json.dumps([m.model_dump(mode="json") for m in sim.get_messages()], indent=2)
            )
            return convert_success(sim, run_id, task.id, args, seed, simswap_commit, gpu_name, transcript_rel)
        except Exception as exc:  # noqa: BLE001 - retried, then recorded, never swallowed
            last_exc = exc
            print(f"  attempt {attempt}/{args.max_retries} failed for {run_id}: {exc}")
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds)

    (transcripts_dir / f"{run_id}.json").write_text(json.dumps([]))
    return convert_failure(run_id, task.id, args, seed, simswap_commit, gpu_name, transcript_rel, last_exc)


class _FakeMessage:
    """Stands in for a tau2 message - just needs model_dump for the transcript writer."""

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text

    def model_dump(self, mode: str = "json") -> dict:
        return {"role": self.role, "text": self.text}


class _FakeRewardInfo:
    """Stands in for tau2 RewardInfo - one read miss, everything else clean."""

    def __init__(self) -> None:
        self.reward = 1.0
        self.partial_action_reward = {
            "read": {"count": 4, "correct": 3},
            "write": {"count": 2, "correct": 2},
        }
        self.info = None


class _FakeSim:
    """Stands in for a tau2 simulation result - no LLM, no GPU, real shape."""

    def __init__(self) -> None:
        self.reward_info = _FakeRewardInfo()
        self.termination_reason = "user_stop"
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.duration = 1.23
        self.agent_cost = 0.0
        self.user_cost = 0.0

    def get_messages(self) -> list[_FakeMessage]:
        return [_FakeMessage("user", "mock turn"), _FakeMessage("assistant", "mock reply")]


def run_dry_run(out: Path, simswap_commit: str) -> None:
    """One fake cell through the real convert/save/log path, no tau2, no GPU."""
    gpu_name = GPU_NAME_DRY_RUN
    transcripts_dir = out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    run_id = "dryrun_task00__dry-run-agent__dry-run-user__seed0"
    run_path = out / f"{run_id}.json"
    transcript_rel = f"transcripts/{run_id}.json"
    fake_args = SimpleNamespace(agent_model="dry-run-agent", agent_revision="dry-run-rev",
                                 agent_quantization="dry-run-quant",
                                 user_model="dry-run-user", user_revision="dry-run-rev",
                                 user_quantization="dry-run-quant",
                                 domain="retail", max_steps=MAX_STEPS)

    if run_path.exists():
        print(f"DRY RUN: {run_path} already on disk, showing it")
        record = RunRecord.load(run_path)
    else:
        sim = _FakeSim()
        (transcripts_dir / f"{run_id}.json").write_text(
            json.dumps([m.model_dump(mode="json") for m in sim.get_messages()], indent=2)
        )
        record = convert_success(sim, run_id, "dryrun_task00", fake_args, 0, simswap_commit, gpu_name, transcript_rel)
        record.save(run_path)
        print(f"DRY RUN: wrote {run_path}")

    print(json.dumps(record.to_dict(), indent=2))
    log_experiment(fake_args, simswap_commit, gpu_name, 1, 0, 0,
                    tau2_version="DRY_RUN", tau2_commit="DRY_RUN")


def log_experiment(args: argparse.Namespace, simswap_commit: str, gpu_name: str,
                    n_completed: int, n_failed: int, n_skipped: int,
                    tau2_version: str = TAU2_VERSION_PINNED, tau2_commit: str = TAU2_COMMIT_PINNED) -> None:
    path = Path("EXPERIMENTS.md")
    header = "| date | simswap_commit | gpu_name | agent_model | agent_revision | user_model | user_revision | tau2_version | tau2_commit | completed | failed | skipped |\n"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    if not path.exists():
        path.write_text("# Experiments\n\n" + header + sep)
    date = time.strftime("%Y-%m-%d")
    row = (f"| {date} | {simswap_commit[:12]} | {gpu_name} | {args.agent_model} | {args.agent_revision} | "
           f"{args.user_model} | {args.user_revision} | {tau2_version} | {tau2_commit[:12]} | "
           f"{n_completed} | {n_failed} | {n_skipped} |\n")
    with open(path, "a") as f:
        f.write(row)


def main() -> None:
    args = parse_args()

    if args.print_tasks:
        print(select_tasks(DOMAIN, N_TASKS))
        return

    simswap_commit = get_simswap_commit()

    if args.dry_run:
        run_dry_run(args.out, simswap_commit)
        return

    check_tau2_version()
    gpu_name = get_gpu_name()

    args.out.mkdir(parents=True, exist_ok=True)
    check_gpu_consistency(args.out, gpu_name)

    from tau2.runner import get_tasks

    task_ids = select_tasks(args.domain, N_TASKS)
    tasks = {t.id: t for t in get_tasks(args.domain, task_ids=task_ids)}

    transcripts_dir = args.out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    n_completed = n_failed = n_skipped = 0

    agent_slug = args.agent_model.replace("/", "-")
    user_slug = args.user_model.replace("/", "-")
    for task_id in task_ids:
        for seed in SEEDS:
            run_id = f"{task_id}__{agent_slug}__{user_slug}__seed{seed}"
            run_path = args.out / f"{run_id}.json"
            if run_path.exists():
                print(f"skip {run_id}: already on disk")
                n_skipped += 1
                continue

            print(f"running {run_id}")
            record = run_cell(args, tasks[task_id], seed, run_id, simswap_commit, gpu_name, transcripts_dir)
            record.save(run_path)
            if record.status == "completed":
                n_completed += 1
            else:
                n_failed += 1
                print(f"  RECORDED AS FAILED after {args.max_retries} attempts: {run_id} ({record.status})")

    print(f"\ndone: completed={n_completed} failed={n_failed} skipped={n_skipped}")
    log_experiment(args, simswap_commit, gpu_name, n_completed, n_failed, n_skipped)


if __name__ == "__main__":
    main()
