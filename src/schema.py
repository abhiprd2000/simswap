"""One run record: one (task, agent, user simulator, seed) episode.

read/write tally shape mirrors tau2-bench's RewardInfo.partial_action_reward
(reward_info.partial_action_reward -> {"read": {"count", "correct"}, "write": {...}}).
Verified field-by-field against real tau2 output at Gate 1, 2026-08-15
(see sweep.py module docstring and EXPERIMENTS.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

STATUSES = ("completed", "error", "timeout")

# our normalized categories, not tau2's raw enum - map tau2 -> this in sweep.py.
# Verified at Gate 1: tau2's TerminationReason.value is lowercase snake_case
# ("max_steps", not "MAX_STEPS") - sweep.py's map is keyed on .value.
TERMINATION_REASONS = (
    "normal",
    "max_turns",
    "user_ended",
    "agent_error",
    "harness_crash",
)


@dataclass
class ActionTally:
    """Count of correct vs total actions of one type (read or write)."""

    count: int = 0
    correct: int = 0

    def __post_init__(self) -> None:
        if self.count < 0 or self.correct < 0:
            raise ValueError(f"negative action tally: {self}")
        if self.correct > self.count:
            raise ValueError(f"correct > count: {self}")

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.count if self.count > 0 else None


@dataclass
class RunRecord:
    schema_version: int
    run_id: str

    # provenance - tau2-bench changed grading across versions (v1.0.1), runs
    # across tau2 versions are not comparable. pin and record, never upgrade
    # mid-sweep.
    tau2_version: str
    tau2_commit: str
    simswap_commit: str  # our repo commit for this run

    # identity
    task_id: str
    domain: str  # retail / telecom / etc
    agent_model: str
    agent_model_revision: str  # exact HF revision hash, not a tag
    user_model: str
    user_model_revision: str  # exact HF revision hash, not a tag
    agent_quantization: str  # e.g. "fp16", "awq-int4" - method actually served
    user_quantization: str
    seed: int

    # config
    temperature: float
    gpu_name: str  # nvidia-smi name, whole sweep must be one GPU model
    timestamp: str  # ISO 8601, UTC
    max_steps: int  # tau2 TextRunConfig cap this run was given, not a default assumed

    # outcome
    status: str  # one of STATUSES - our harness-level result
    termination_reason: str  # one of TERMINATION_REASONS - why the episode ended
    error: str | None  # null on success, traceback string on failure
    task_success: bool | None  # binary metric, SECONDARY only
    reward: float | None  # raw tau2 reward
    reward_info_note: dict | None  # tau2 RewardInfo.info - free-text debug note, e.g. why reward is 0
    n_turns: int  # control variable - some simulators talk longer

    # action-level correctness — PRIMARY METRIC
    read: ActionTally
    write: ActionTally

    # extras
    wall_clock_seconds: float | None
    agent_cost: float | None
    user_cost: float | None
    transcript_path: str  # full conversation on disk, non-negotiable

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"bad status {self.status!r}, must be one of {STATUSES}")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"bad termination_reason {self.termination_reason!r}, "
                f"must be one of {TERMINATION_REASONS}"
            )
        if isinstance(self.read, dict):
            self.read = ActionTally(**self.read)
        if isinstance(self.write, dict):
            self.write = ActionTally(**self.write)
        if self.status == "completed":
            if self.reward is None:
                raise ValueError(f"completed run {self.run_id} has no reward")
            if self.error is not None:
                raise ValueError(f"completed run {self.run_id} has an error set")
        else:
            if not self.error:
                raise ValueError(f"non-completed run {self.run_id} has no error")
        if not self.transcript_path:
            raise ValueError(f"run {self.run_id} has no transcript_path")
        if not self.gpu_name:
            raise ValueError(f"run {self.run_id} has no gpu_name")
        if self.n_turns < 0:
            raise ValueError(f"run {self.run_id} has negative n_turns")
        if self.max_steps <= 0:
            raise ValueError(f"run {self.run_id} has non-positive max_steps")
        if not self.agent_quantization or not self.user_quantization:
            raise ValueError(f"run {self.run_id} missing quantization method")

    @property
    def action_total(self) -> int:
        return self.read.count + self.write.count

    @property
    def action_correct(self) -> int:
        return self.read.correct + self.write.correct

    @property
    def action_accuracy(self) -> float | None:
        total = self.action_total
        return self.action_correct / total if total > 0 else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        return cls.from_dict(json.loads(path.read_text()))
