"""Single parameterized Kaggle kernel entry point for a Gate 1 (agent, user) run.

Reads its configuration from environment variables (set via kernel-metadata.json's
"environment_variables", or Kaggle's kernel UI) so one script reproduces every run in
this study - point it at a different config instead of writing a new script per run.
Requires the `simswap-code` dataset (src/) attached as a kernel input.

Environment variables (all optional, defaults reproduce the Qwen self-play run):
  SIMSWAP_RUN_NAME               base name for output files (default "gate1_run")
  SIMSWAP_AGENT_MODEL             HF repo id (default "Qwen/Qwen2.5-7B-Instruct-AWQ")
  SIMSWAP_AGENT_PARSER             vLLM --tool-call-parser value (default "hermes")
  SIMSWAP_AGENT_CHAT_TEMPLATE_SUBSTR   substring to find a chat template in the vLLM
                                    source checkout, e.g. "llama3.1_json" (default: none)
  SIMSWAP_AGENT_QUANT_LABEL        free-text label recorded in the run record (default "awq-int4")
  SIMSWAP_AGENT_TRUST_REMOTE_CODE  "1" to pass --trust-remote-code for the agent (default "0")
  SIMSWAP_USER_MODEL / SIMSWAP_USER_PARSER / SIMSWAP_USER_CHAT_TEMPLATE_SUBSTR /
  SIMSWAP_USER_QUANT_LABEL / SIMSWAP_USER_TRUST_REMOTE_CODE   same, for the user side
  SIMSWAP_MAX_MODEL_LEN            vLLM max_model_len for both servers (default 32768)
  SIMSWAP_MAX_STEPS                tau2 TextRunConfig step cap (default sweep.MAX_STEPS)
  SIMSWAP_TASK_INDICES             comma-separated task indices, e.g. "0,1,3" (default "0")

See kernels/README.md for the exact configs used for this study's final data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CANDIDATES = [
    "/kaggle/input/simswap-code",
    "/kaggle/input/datasets/<KAGGLE_USERNAME>/simswap-code",
]
CODE_DIR = None
for _ in range(30):
    for c in CANDIDATES:
        if os.path.exists(os.path.join(c, "gate1_core.py")):
            CODE_DIR = c
            break
    if CODE_DIR:
        break
    time.sleep(2)
if CODE_DIR is None:
    raise RuntimeError(f"gate1_core.py not found under any of {CANDIDATES}")
sys.path.insert(0, CODE_DIR)
import gate1_core
import sweep


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_task_indices(name: str, default: list[int]) -> list[int]:
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default


RUN_NAME = env("SIMSWAP_RUN_NAME", "gate1_run")
AGENT_MODEL = env("SIMSWAP_AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
AGENT_PARSER = env("SIMSWAP_AGENT_PARSER", "hermes")
AGENT_CHAT_TEMPLATE_SUBSTR = os.environ.get("SIMSWAP_AGENT_CHAT_TEMPLATE_SUBSTR")
AGENT_QUANT_LABEL = env("SIMSWAP_AGENT_QUANT_LABEL", "awq-int4")
AGENT_TRUST_REMOTE_CODE = env("SIMSWAP_AGENT_TRUST_REMOTE_CODE", "0") == "1"

USER_MODEL = env("SIMSWAP_USER_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
USER_PARSER = env("SIMSWAP_USER_PARSER", "hermes")
USER_CHAT_TEMPLATE_SUBSTR = os.environ.get("SIMSWAP_USER_CHAT_TEMPLATE_SUBSTR")
USER_QUANT_LABEL = env("SIMSWAP_USER_QUANT_LABEL", "awq-int4")
USER_TRUST_REMOTE_CODE = env("SIMSWAP_USER_TRUST_REMOTE_CODE", "0") == "1"

MAX_MODEL_LEN = int(env("SIMSWAP_MAX_MODEL_LEN", "32768"))
MAX_STEPS = int(env("SIMSWAP_MAX_STEPS", str(sweep.MAX_STEPS)))
TASK_INDICES = env_task_indices("SIMSWAP_TASK_INDICES", [0])

gate1_core.install_deps()


def resolve_template(substr: str | None) -> str | None:
    if not substr:
        return None
    template = gate1_core.find_chat_template(substr)
    if template is None:
        raise RuntimeError(f"no chat template matching {substr!r} found in the vLLM source checkout")
    return template


agent_chat_template = resolve_template(AGENT_CHAT_TEMPLATE_SUBSTR)
user_chat_template = resolve_template(USER_CHAT_TEMPLATE_SUBSTR)

summary = []
for idx in TASK_INDICES:
    run_name = f"{RUN_NAME}_task{idx}"
    try:
        result = gate1_core.run_pair(
            run_name=run_name,
            agent_model=AGENT_MODEL, agent_parser=AGENT_PARSER,
            agent_quantization=None, agent_quantization_label=AGENT_QUANT_LABEL,
            user_model=USER_MODEL, user_parser=USER_PARSER,
            user_quantization=None, user_quantization_label=USER_QUANT_LABEL,
            max_model_len=MAX_MODEL_LEN, max_steps=MAX_STEPS,
            agent_chat_template=agent_chat_template, user_chat_template=user_chat_template,
            agent_trust_remote_code=AGENT_TRUST_REMOTE_CODE, user_trust_remote_code=USER_TRUST_REMOTE_CODE,
            task_index=idx,
        )
        summary.append({"task_index": idx, "run_name": run_name, "status": "ran", "result": result})
    except Exception as e:
        gate1_core.log(f"task_index={idx} raised uncaught: {e}")
        summary.append({"task_index": idx, "run_name": run_name, "status": "uncaught-exception", "error": str(e)})

(gate1_core.WORK / f"{RUN_NAME}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
gate1_core.log("=== DONE ===\n" + json.dumps(summary, indent=2, default=str))
