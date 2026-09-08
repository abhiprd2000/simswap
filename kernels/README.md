# Kaggle kernel

`gate1_run.py` is the one script that runs every (agent, user, task-set) configuration in
this study. Deploy by uploading `src/` as a Kaggle dataset (`dataset-metadata.template.json`
is a template - copy it, fill in `<KAGGLE_USERNAME>`, and run `kaggle datasets create` from
a directory containing `src/`'s contents), then create a kernel from `gate1_run.py` using
`kernel-metadata.template.json` as a template (fill in `<KAGGLE_USERNAME>` in both the
`id` and `dataset_sources` fields, and edit `environment_variables` for the config you want).

## Exact configs used for this study's final data (`data/runs/`, `data/raw/`)

**Qwen self-play** (`data/{runs,raw}/qwen_task{0,1,2,3}.json`):
```json
{
  "SIMSWAP_RUN_NAME": "qwen",
  "SIMSWAP_AGENT_MODEL": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "SIMSWAP_AGENT_PARSER": "hermes",
  "SIMSWAP_USER_MODEL": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "SIMSWAP_USER_PARSER": "hermes",
  "SIMSWAP_MAX_MODEL_LEN": "32768",
  "SIMSWAP_TASK_INDICES": "0,1,2,3"
}
```

**Llama agent vs. the same Qwen2.5-7B-Instruct-AWQ user** (`data/{runs,raw}/llama_task{0,1,3}.json`
- task 2 excluded, see `docs/SCHEMA_CONFORMANCE.md`):
```json
{
  "SIMSWAP_RUN_NAME": "llama_vs_qwen7b",
  "SIMSWAP_AGENT_MODEL": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
  "SIMSWAP_AGENT_PARSER": "llama3_json",
  "SIMSWAP_AGENT_CHAT_TEMPLATE_SUBSTR": "llama3.1_json",
  "SIMSWAP_USER_MODEL": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "SIMSWAP_USER_PARSER": "hermes",
  "SIMSWAP_MAX_MODEL_LEN": "32768",
  "SIMSWAP_TASK_INDICES": "0,1,3"
}
```

Both used the full `max_steps` cap (`sweep.MAX_STEPS`, unset in the configs above since
that's the default), seed 0, temperature 0.0, on a single Tesla T4 per model (2 GPUs per
kernel). See `docs/FAILURE_ANALYSIS.md` for why the Llama run above (a matched user
simulator) supersedes an earlier run against a smaller, fixed user model.

## Output

Each task index writes `converted_record_<run_name>_task<N>.json` (the run record) and
`raw_tau2_dump_<run_name>_task<N>.json` (the full transcript) to `/kaggle/working/`. Pull
both, rename to `<agent>_task<N>.json`, and place under `data/runs/` and `data/raw/`
respectively to reproduce this repository's `data/` layout.
