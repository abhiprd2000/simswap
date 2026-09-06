"""Gate 1 diagnostic: one task, one agent, one user model, seed 0.

Pass A: Qwen2.5-1.5B-Instruct both sides. Field diff, ignore score.
Pass B: Qwen2.5-7B-Instruct both sides. Real VRAM/wall-clock/OOM numbers.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
AGENT_PORT, USER_PORT = 8001, 8002


def _log_path() -> Path:
    return WORK / "gate1_log.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_log_path(), "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    log(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    log(f"  rc={r.returncode}\nSTDOUT:\n{r.stdout[-4000:]}\nSTDERR:\n{r.stderr[-4000:]}")
    return r


def safe_dump(obj) -> object:
    """Best-effort raw dump: pydantic model_dump, else __dict__, else str."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception as e:
            return f"<model_dump failed: {e}>"
    if hasattr(obj, "__dict__"):
        out = {}
        for k, v in vars(obj).items():
            try:
                json.dumps(v, default=str)
                out[k] = v
            except Exception:
                out[k] = repr(v)
        return out
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def vram_monitor(stop_event: threading.Event, vram_csv: Path) -> None:
    with open(vram_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "gpu_index", "mem_used_mib"])
        while not stop_event.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            ts = time.time()
            for line in r.stdout.strip().splitlines():
                idx, used = [x.strip() for x in line.split(",")]
                w.writerow([ts, idx, used])
            f.flush()
            time.sleep(2)


def wait_for_server(port: int, timeout: int = 900) -> bool:
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    log(f"server on port {port} is up after {time.time()-start:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_vllm(gpu: int, port: int, model: str, max_model_len: int | None,
                quantization: str | None = None, tool_call_parser: str = "hermes",
                chat_template: str | None = None, trust_remote_code: bool = False,
                log_prefix: str = "") -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    args = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--dtype", "float16", "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        # tau2's agent sends tool_choice="auto" - vLLM 400s on that
        # without these. Parser is family-specific - see MODEL_CANDIDATES.md.
        "--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser,
    ]
    if max_model_len is not None:
        args += ["--max-model-len", str(max_model_len)]
    if quantization is not None:
        # exact vLLM --quantization value, e.g. "awq" - Turing (T4, SM75) has
        # no Marlin kernel, so this must NOT be left to vLLM's auto-detect if
        # that would pick awq_marlin. Verify against MODEL_CANDIDATES.md.
        args += ["--quantization", quantization]
    if chat_template is not None:
        args += ["--chat-template", chat_template]
    if trust_remote_code:
        # e.g. internlm2_5-7b-chat-4bit: vLLM refuses to load without this -
        # "repository contains custom code which must be executed."
        args += ["--trust-remote-code"]
    logfile = open(WORK / f"{log_prefix}vllm_gpu{gpu}_port{port}.log", "w")
    log(f"starting vllm: gpu={gpu} port={port} model={model} max_model_len={max_model_len} "
        f"quantization={quantization} parser={tool_call_parser} chat_template={chat_template} "
        f"trust_remote_code={trust_remote_code} cmd={' '.join(args)}")
    return subprocess.Popen(args, env=env, stdout=logfile, stderr=subprocess.STDOUT)


def peak_vram_per_gpu(vram_csv: Path) -> dict:
    peaks: dict[str, int] = {}
    if not vram_csv.exists():
        return peaks
    with open(vram_csv) as f:
        for row in csv.DictReader(f):
            g, m = row["gpu_index"], int(row["mem_used_mib"])
            peaks[g] = max(peaks.get(g, 0), m)
    return peaks


def run(pass_name: str, model: str, max_model_len: int | None,
        quantization: str | None = None, quantization_label: str = "fp16") -> None:
    """quantization: vLLM --quantization value (None = let vLLM auto-detect/fp16).
    quantization_label: what gets recorded in the run record (schema requires
    a non-empty string), e.g. "fp16" or "awq-int4"."""
    WORK.mkdir(parents=True, exist_ok=True)
    vram_csv = WORK / f"vram_timeline_pass{pass_name}.csv"
    log(f"=== GATE 1 PASS {pass_name} === model={model} max_model_len={max_model_len} "
        f"quantization={quantization} quantization_label={quantization_label}")

    sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
    # tau2-bench isn't on PyPI under that name - "tau2" is, but 1.0.1 was
    # never released there, only at this pinned commit. Domain task data
    # (data/tau2/domains/retail/tasks.json etc) lives in the repo, not the
    # wheel - a plain `pip install git+...` loses it (FileNotFoundError
    # at runtime), and an editable install from /kaggle/working left
    # `import tau2` broken for unknown reasons (also polluted kernel
    # output with the whole cloned repo, since /kaggle/working is the
    # output dir). Regular (non-editable) install from a /tmp clone +
    # TAU2_DATA_DIR, per docs/getting-started.md's documented path for
    # non-editable installs.
    TAU2_SRC = "/tmp/tau2-bench-src"
    sh(f"rm -rf {TAU2_SRC}")
    sh(f"git clone --quiet https://github.com/sierra-research/tau2-bench.git {TAU2_SRC}")
    sh(f"cd {TAU2_SRC} && git checkout --quiet fc0055dc4e0a316c3f83133267fbd6faaa770992")
    sh(f"{sys.executable} -m pip install -q {TAU2_SRC}")
    os.environ["TAU2_DATA_DIR"] = f"{TAU2_SRC}/data"
    sh(f"{sys.executable} -m pip show tau2")
    sh(f"{sys.executable} -c \"import tau2; print('tau2 module at', tau2.__file__)\"")
    # pinned, not -U: Gate 1 Pass A/B both ran clean on 0.27.1 (see
    # kaggle_output_a/vllm_gpu0_port8001.log). MODEL_CANDIDATES.md found
    # unrelated T4/SM75 regressions in other vLLM releases (0.15.1 crashes
    # at init, 0.17.0 hangs, 0.24.0+ drops FlashInfer on SM75) - don't
    # gamble on whatever "latest" resolves to on a later run.
    sh(f"{sys.executable} -m pip install -q vllm==0.27.1 xformers")

    import schema  # noqa: F401,E402
    import sweep  # noqa: E402

    sweep.check_tau2_version()
    gpu_name = sweep.get_gpu_name()
    log(f"gpu_name={gpu_name}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    result = {"pass": pass_name, "model": model, "max_model_len": max_model_len, "gpu_name": gpu_name}
    agent_proc = user_proc = None
    t_start = time.time()
    try:
        agent_proc = start_vllm(0, AGENT_PORT, model, max_model_len, quantization)
        user_proc = start_vllm(1, USER_PORT, model, max_model_len, quantization)

        agent_up = wait_for_server(AGENT_PORT)
        user_up = wait_for_server(USER_PORT)
        result["agent_server_up"] = agent_up
        result["user_server_up"] = user_up
        if not (agent_up and user_up):
            for proc, name in ((agent_proc, "agent"), (user_proc, "user")):
                if proc.poll() is not None:
                    log(f"{name} server exited early with code {proc.returncode}")
            raise RuntimeError("vLLM server(s) failed to come up - see vllm_gpu*.log for OOM/crash")

        from tau2.runner import get_tasks, run_single_task
        from tau2.data_model.simulation import TextRunConfig

        tasks = get_tasks("retail", task_split_name="base")
        task = sorted(tasks, key=lambda t: (int(t.id) if t.id.isdigit() else 0, t.id))[0]
        result["task_id"] = task.id
        log(f"fixed task_id={task.id}")

        fake_args = type("Args", (), {
            "domain": "retail", "agent_model": model, "agent_revision": "gate1-live",
            "user_model": model, "user_revision": "gate1-live", "llm_timeout_seconds": 300.0,
            "agent_quantization": quantization_label, "user_quantization": quantization_label,
            "max_steps": sweep.MAX_STEPS,
        })()

        config = TextRunConfig(
            domain="retail", agent="llm_agent", user="user_simulator",
            llm_agent=f"openai/{model}",
            llm_args_agent={"temperature": 0.0, "timeout": 300.0,
                             "api_base": f"http://localhost:{AGENT_PORT}/v1", "api_key": "EMPTY"},
            llm_user=f"openai/{model}",
            llm_args_user={"temperature": 0.0, "timeout": 300.0,
                            "api_base": f"http://localhost:{USER_PORT}/v1", "api_key": "EMPTY"},
            seed=0,
            max_steps=sweep.MAX_STEPS,
        )

        run_start = time.time()
        sim = run_single_task(config, task, seed=0)
        run_wall = time.time() - run_start
        result["measured_wall_clock_seconds"] = run_wall
        log(f"run_single_task done in {run_wall:.1f}s")

        reward_info = sim.reward_info
        raw = {
            "termination_reason_raw": {
                "value": getattr(sim.termination_reason, "value", sim.termination_reason),
                "type": type(sim.termination_reason).__name__,
            },
            "sim_dump": safe_dump(sim),
            "reward_info_dump": safe_dump(reward_info),
            "partial_action_reward_raw": (
                json.loads(json.dumps(reward_info.partial_action_reward, default=str))
                if reward_info is not None else None
            ),
            "n_messages": len(sim.get_messages()),
            "duration": sim.duration,
            "agent_cost": sim.agent_cost,
            "user_cost": sim.user_cost,
            "timestamp": str(sim.timestamp),
            "trial": getattr(sim, "trial", None),
        }
        (WORK / f"raw_tau2_dump_pass{pass_name}.json").write_text(json.dumps(raw, indent=2, default=str))
        log(f"wrote raw_tau2_dump_pass{pass_name}.json")

        record = sweep.convert_success(sim, f"gate1_{pass_name}", task.id, fake_args, 0,
                                        "gate1-diagnostic", gpu_name, "gate1_transcript.json")
        (WORK / f"converted_record_pass{pass_name}.json").write_text(json.dumps(record.to_dict(), indent=2))
        log(f"wrote converted_record_pass{pass_name}.json - schema validation passed")

        result["converted_ok"] = True
        result["mapped_termination_reason"] = record.termination_reason
        result["reward"] = record.reward
        result["n_turns"] = record.n_turns
        result["read"] = {"count": record.read.count, "correct": record.read.correct}
        result["write"] = {"count": record.write.count, "correct": record.write.correct}
        result["wall_clock_seconds_recorded"] = record.wall_clock_seconds

    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
        result["error"] = traceback.format_exc()
        result["converted_ok"] = False
    finally:
        result["total_wall_clock_seconds"] = time.time() - t_start
        time.sleep(3)  # let vram monitor grab post-load peak
        stop_event.set()
        mon.join(timeout=5)
        result["peak_vram_mib_per_gpu"] = peak_vram_per_gpu(vram_csv)

        for proc in (agent_proc, user_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()

        sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")

        (WORK / f"gate1_result_pass{pass_name}.json").write_text(json.dumps(result, indent=2, default=str))
        log(f"=== DONE PASS {pass_name} ===\n{json.dumps(result, indent=2, default=str)}")


VLLM_SRC = "/tmp/vllm-src"


def install_deps() -> None:
    """One-time setup for a kernel session: tau2 (pinned commit) + vllm==0.27.1
    + xformers + a vLLM source checkout (for tool-call chat-template jinja
    files, which don't ship in the wheel). Call once, then run_pair() as many
    times as needed in the same session."""
    WORK.mkdir(parents=True, exist_ok=True)
    sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
    TAU2_SRC = "/tmp/tau2-bench-src"
    sh(f"rm -rf {TAU2_SRC}")
    sh(f"git clone --quiet https://github.com/sierra-research/tau2-bench.git {TAU2_SRC}")
    sh(f"cd {TAU2_SRC} && git checkout --quiet fc0055dc4e0a316c3f83133267fbd6faaa770992")
    sh(f"{sys.executable} -m pip install -q {TAU2_SRC}")
    os.environ["TAU2_DATA_DIR"] = f"{TAU2_SRC}/data"
    sh(f"{sys.executable} -m pip show tau2")

    # pinned, not -U: Gate 1 Pass A/B both ran clean on 0.27.1 (see
    # kaggle_output_a/vllm_gpu0_port8001.log). MODEL_CANDIDATES.md found
    # unrelated T4/SM75 regressions in other vLLM releases (0.15.1 crashes
    # at init, 0.17.0 hangs, 0.24.0+ drops FlashInfer on SM75).
    sh(f"{sys.executable} -m pip install -q vllm==0.27.1 xformers")

    sh(f"rm -rf {VLLM_SRC}")
    r = sh(f"git clone --quiet --depth 1 --branch v0.27.1 https://github.com/vllm-project/vllm.git {VLLM_SRC}")
    if r.returncode != 0:
        log(f"WARNING: vLLM source clone at tag v0.27.1 failed (rc={r.returncode}), "
            "chat-template lookups will fail for families that need one.")

    patch_evaluator_crash_guard()


def patch_evaluator_crash_guard() -> None:
    """tau2.runner.simulation.run_simulation does `simulation =
    orchestrator.run()` (fully populates the transcript), THEN separately
    calls `evaluate_simulation(simulation=simulation, ...)` for the reward.
    If that second call raises - e.g. a task's nl_assertions check tries to
    call a real judge LLM via litellm and we haven't configured one, so it
    falls through to litellm's OpenAI default and hits a missing
    OPENAI_API_KEY - the exception propagates uncaught and the WHOLE
    `simulation` object, transcript included, is lost. Confirmed against
    the real source at the pinned commit: run_simulation has no try/except
    around the evaluate_simulation call, and the exception is not caught
    anywhere in the run_single_task -> run_simulation chain either.

    Fix: wrap evaluate_simulation so a crash there produces a stub
    RewardInfo (reward=0.0, everything else None/default, info explains
    the crash) instead of losing the transcript - our own
    sweep.convert_success() only needs reward_info to be non-None with a
    real .reward/.partial_action_reward, both satisfied by the stub.

    Must patch tau2.runner.simulation.evaluate_simulation specifically,
    NOT tau2.evaluator.evaluator.evaluate_simulation - run_simulation
    imported it via `from tau2.evaluator.evaluator import
    evaluate_simulation`, which bound a name in tau2.runner.simulation's
    OWN namespace at import time. Patching the origin module doesn't reach
    that already-bound reference."""
    import tau2.runner.simulation as tau2_sim
    from tau2.data_model.simulation import RewardInfo

    original = tau2_sim.evaluate_simulation

    def guarded(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            log(f"WARNING: evaluate_simulation crashed, using stub RewardInfo "
                f"so the transcript isn't lost: {exc}")
            return RewardInfo(reward=0.0, info={"error": f"evaluation crashed: {exc}"})

    tau2_sim.evaluate_simulation = guarded
    log("patched tau2.runner.simulation.evaluate_simulation with crash guard")


def find_chat_template(basename_substr: str) -> str | None:
    """Locate a tool-call chat-template jinja file inside the cloned vLLM
    source by substring match on filename - path within examples/ has moved
    across vLLM releases, don't hardcode it."""
    r = subprocess.run(
        ["find", VLLM_SRC, "-iname", f"*{basename_substr}*.jinja"],
        capture_output=True, text=True,
    )
    hits = [line for line in r.stdout.splitlines() if line.strip()]
    log(f"find_chat_template({basename_substr!r}) -> {hits}")
    return hits[0] if hits else None


def count_tool_calls(messages, role: str = "assistant") -> int:
    """Count messages with >=1 REAL structured tool call (parsed by vLLM's
    tool-call parser), not the model just writing tool-call-shaped plain
    text - Gate 1 Pass A's 1.5B model did exactly that and it must not
    count."""
    n = 0
    for m in messages:
        d = m.model_dump(mode="json") if hasattr(m, "model_dump") else m
        if d.get("role") == role and d.get("tool_calls"):
            n += 1
    return n


def run_pair(run_name: str,
             agent_model: str, agent_parser: str,
             agent_quantization: str | None, agent_quantization_label: str,
             user_model: str, user_parser: str,
             user_quantization: str | None, user_quantization_label: str,
             max_model_len: int | None, max_steps: int,
             agent_chat_template: str | None = None,
             user_chat_template: str | None = None,
             agent_trust_remote_code: bool = False,
             user_trust_remote_code: bool = False,
             task_index: int = 0) -> dict:
    """One (agent, user) episode on task '0', general enough for both the
    smoke test (distinct agent per candidate, fixed small user) and the
    Pass B AWQ redo (same AWQ model both sides). Call install_deps() first,
    once per kernel session - this does not reinstall anything."""
    import schema  # noqa: F401
    import sweep

    vram_csv = WORK / f"vram_timeline_{run_name}.csv"
    log(f"=== RUN {run_name} === agent={agent_model} (parser={agent_parser}, quant={agent_quantization}) "
        f"user={user_model} (parser={user_parser}, quant={user_quantization}) max_steps={max_steps}")

    gpu_name = sweep.get_gpu_name()
    log(f"gpu_name={gpu_name}")

    stop_event = threading.Event()
    mon = threading.Thread(target=vram_monitor, args=(stop_event, vram_csv), daemon=True)
    mon.start()

    result = {
        "run_name": run_name, "agent_model": agent_model, "user_model": user_model,
        "agent_parser": agent_parser, "user_parser": user_parser,
        "agent_quantization_label": agent_quantization_label,
        "user_quantization_label": user_quantization_label,
        "max_model_len": max_model_len, "max_steps": max_steps, "gpu_name": gpu_name,
    }
    agent_proc = user_proc = None
    t_start = time.time()
    try:
        agent_proc = start_vllm(0, AGENT_PORT, agent_model, max_model_len, agent_quantization,
                                 agent_parser, agent_chat_template, agent_trust_remote_code,
                                 log_prefix=f"{run_name}_")
        user_proc = start_vllm(1, USER_PORT, user_model, max_model_len, user_quantization,
                                user_parser, user_chat_template, user_trust_remote_code,
                                log_prefix=f"{run_name}_")

        agent_up = wait_for_server(AGENT_PORT)
        user_up = wait_for_server(USER_PORT)
        result["agent_server_up"] = agent_up
        result["user_server_up"] = user_up
        if not (agent_up and user_up):
            for proc, name in ((agent_proc, "agent"), (user_proc, "user")):
                if proc.poll() is not None:
                    log(f"{name} server exited early with code {proc.returncode}")
            raise RuntimeError("vLLM server(s) failed to come up - see vllm_gpu*.log for OOM/crash")

        from tau2.runner import get_tasks, run_single_task
        from tau2.data_model.simulation import TextRunConfig

        tasks = get_tasks("retail", task_split_name="base")
        sorted_tasks = sorted(tasks, key=lambda t: (int(t.id) if t.id.isdigit() else 0, t.id))
        task = sorted_tasks[task_index]
        result["task_id"] = task.id
        log(f"task_index={task_index} task_id={task.id}")

        fake_args = type("Args", (), {
            "domain": "retail", "agent_model": agent_model, "agent_revision": "gate1-live",
            "user_model": user_model, "user_revision": "gate1-live", "llm_timeout_seconds": 300.0,
            "agent_quantization": agent_quantization_label, "user_quantization": user_quantization_label,
            "max_steps": max_steps,
        })()

        config = TextRunConfig(
            domain="retail", agent="llm_agent", user="user_simulator",
            llm_agent=f"openai/{agent_model}",
            llm_args_agent={"temperature": 0.0, "timeout": 300.0,
                             "api_base": f"http://localhost:{AGENT_PORT}/v1", "api_key": "EMPTY"},
            llm_user=f"openai/{user_model}",
            llm_args_user={"temperature": 0.0, "timeout": 300.0,
                            "api_base": f"http://localhost:{USER_PORT}/v1", "api_key": "EMPTY"},
            seed=0,
            max_steps=max_steps,
        )

        run_start = time.time()
        sim = run_single_task(config, task, seed=0)
        run_wall = time.time() - run_start
        result["measured_wall_clock_seconds"] = run_wall
        log(f"run_single_task done in {run_wall:.1f}s")

        messages = sim.get_messages()
        result["n_turns"] = len(messages)
        result["agent_tool_calls"] = count_tool_calls(messages, role="assistant")
        log(f"n_turns={result['n_turns']} agent_tool_calls={result['agent_tool_calls']}")

        reward_info = sim.reward_info
        raw = {
            "termination_reason_raw": {
                "value": getattr(sim.termination_reason, "value", sim.termination_reason),
                "type": type(sim.termination_reason).__name__,
            },
            "sim_dump": safe_dump(sim),
            "reward_info_dump": safe_dump(reward_info),
            "n_messages": len(messages),
            "duration": sim.duration,
            "timestamp": str(sim.timestamp),
        }
        (WORK / f"raw_tau2_dump_{run_name}.json").write_text(json.dumps(raw, indent=2, default=str))
        log(f"wrote raw_tau2_dump_{run_name}.json")

        transcript_name = f"{run_name}_transcript.json"
        (WORK / transcript_name).write_text(
            json.dumps([m.model_dump(mode="json") for m in messages], indent=2, default=str)
        )
        log(f"wrote {transcript_name}")

        record = sweep.convert_success(sim, run_name, task.id, fake_args, 0,
                                        "gate1-diagnostic", gpu_name, transcript_name)
        (WORK / f"converted_record_{run_name}.json").write_text(json.dumps(record.to_dict(), indent=2))
        log(f"wrote converted_record_{run_name}.json - schema validation passed")

        result["converted_ok"] = True
        result["mapped_termination_reason"] = record.termination_reason
        result["reward"] = record.reward
        result["wall_clock_seconds_recorded"] = record.wall_clock_seconds

    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
        result["error"] = traceback.format_exc()
        result["converted_ok"] = False
    finally:
        result["total_wall_clock_seconds"] = time.time() - t_start
        time.sleep(3)  # let vram monitor grab post-load peak
        stop_event.set()
        mon.join(timeout=5)
        result["peak_vram_mib_per_gpu"] = peak_vram_per_gpu(vram_csv)

        for proc in (agent_proc, user_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        for proc in (agent_proc, user_proc):
            if proc is not None:
                try:
                    proc.wait(timeout=20)
                except Exception:
                    proc.kill()

        sh("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")

        (WORK / f"result_{run_name}.json").write_text(json.dumps(result, indent=2, default=str))
        log(f"=== DONE {run_name} ===\n{json.dumps(result, indent=2, default=str)}")

    return result
