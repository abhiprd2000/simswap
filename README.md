# simswap

Tool-call grounding failures in tau2-bench agents differ by model in both rate and
provenance, and the contrast survives holding the user simulator fixed across agents. See
`paper/PAPER.md` for the full writeup; this README reproduces every number in it.

Anonymized for double-blind review: no author names, emails, or account handles appear
anywhere in this repository. Kaggle kernel/dataset ids in `kernels/` use a
`<KAGGLE_USERNAME>` placeholder - fill in your own before deploying.

## Layout

```
ANALYSIS_PLAN.md              pre-registered design (untouched after real data landed)
ANALYSIS_PLAN_AMENDMENT.md    what changed from the plan, and why
src/                          all analysis and harness code
tests/                        regression tests for the grounding/schema logic
kernels/                      the one parameterized Kaggle kernel script
data/runs/                    final run records (7 conversations) used in the paper
data/raw/                     full tau2 transcripts for those same 7 conversations
tables/                       CSV tables cited in ANALYSIS_PLAN*.md / the paper
docs/                         methodology writeups: grounding, schema conformance,
                               the audit history, and the regression-test rationale
paper/                        PAPER.md (source) and main.tex (NeurIPS 2026 template)
```

## Setup

```
pip install -r requirements.txt
```

`vllm==0.27.1` is only needed to reproduce the GPU runs (see "Reproducing the raw runs"
below) - everything else in this README runs on CPU from the JSON files already in `data/`.

## Reproducing the paper's tables (no GPU needed)

Run from the repository root, in this order:

```
python tests/test_failure_analysis.py       # 11/11 regression tests must pass
python src/failure_analysis.py              # writes failure_analysis_raw.json - Table 1 (grounding),
                                             # Table 3 (provenance), Table 5 (post-error next action)
python src/matched_comparison.py            # pools the above over tasks {0,1,3} - Table 1's headline numbers
python src/schema_conformance.py            # Table 2 (schema conformance)
python src/permutation_test.py              # the exact paired permutation test in §3.1
python src/user_id_mentions.py              # Table 4 (mechanism) - Qwen side only, see note below
```

Each script prints its numbers and writes a `*_raw.json` file (gitignored, regenerate on
demand - not meant to be committed). `src/matched_comparison.py` and `src/permutation_test.py`
both read `failure_analysis_raw.json`, so run `src/failure_analysis.py` first.

**Note on `src/user_id_mentions.py`:** Table 4 compares how many identifiers two
*different* user simulators state (Qwen2.5-7B-Instruct-AWQ vs. the smaller
Qwen2.5-1.5B-Instruct used in an earlier, superseded Llama run) - that comparison is the
entire point of the mechanism finding. The three `llama_task{0,1,3}_1.5b_user.json` files
in `data/raw/` and `data/runs/` are that earlier run, restored from archive; see
`docs/FAILURE_ANALYSIS.md`'s "CONFOUND CHECK" section for why it's superseded as the
headline number but still cited for the mechanism finding.

**Two Llama run sets, don't mix them up:** `data/{raw,runs}/llama_task{0,1,3}.json`
(matched simulator, Qwen2.5-7B-Instruct-AWQ user) feed Sections 3.1-3.2 and the paper's
headline 93.3%/58.4% numbers. `data/{raw,runs}/llama_task{0,1,3}_1.5b_user.json`
(Qwen2.5-1.5B-Instruct user, the original confounded design) feed only Table 4 and the
Section 3.3 confound discussion - do not use them for anything else.

## Reproducing the raw GPU runs

`kernels/gate1_run.py` is the one Kaggle kernel script behind every run in `data/`.
`kernels/README.md` has the exact two configurations (environment variables) used for the
paper's final data, and the deployment steps (upload `src/` as a Kaggle dataset, create a
kernel from `gate1_run.py` with a filled-in `kernel-metadata.template.json`).

**Non-Kaggle path**: start two vLLM servers by hand (one per GPU) and drive `sweep.py`
directly. Both paths need tau2-bench installed from the pinned commit - see
`requirements.txt`.

```
# one per GPU, fp16, tool-choice flags required (tau2's agent sends tool_choice="auto",
# which vLLM 400s on without them) - "hermes" matches Qwen2.5's tool-call format,
# use "llama3_json" + the matching chat template for Llama
CUDA_VISIBLE_DEVICES=0 vllm serve <agent-model> --dtype float16 --port 8001 \
    --enable-auto-tool-choice --tool-call-parser hermes
CUDA_VISIBLE_DEVICES=1 vllm serve <user-model>  --dtype float16 --port 8002 \
    --enable-auto-tool-choice --tool-call-parser hermes

python src/sweep.py \
    --agent-model <served-name> --agent-revision <hf-commit> --agent-url http://localhost:8001/v1 --agent-quantization fp16 \
    --user-model <served-name>  --user-revision <hf-commit>  --user-url http://localhost:8002/v1 --user-quantization fp16
```

Re-running the same command resumes: it skips any (task, seed) cell that already has a
run file on disk. A 7B model does not fit one T4 in fp16 (weights alone exceed a
90%-of-16GB budget before KV cache) - quantize or pick a smaller model.

## Detectability sweep (design justification, not paper data)

`ANALYSIS_PLAN.md`'s minimum-detectable-effect number, and the follow-up numbers in
`ANALYSIS_PLAN_AMENDMENT.md`, come from a synthetic-data sweep, independent of any real
run:

```
python src/fake_runs.py --out runs                 # generate synthetic run records
python src/analyze.py --runs-dir runs               # the real analysis pipeline, on fake data
python src/detectability.py                         # sweeps grid size x planted effect size,
                                                      # writes tables/detectability*.csv
```

`src/fake_runs.py --null` generates noise-only data (no planted effect) as a negative
control - `src/analyze.py` must not report a finding there.

## Paper

`paper/PAPER.md` is the source (Markdown). `paper/main.tex` is the NeurIPS 2026 template
version, same content. Every number in both traces to a file in this repository via the
commands above; `paper/PAPER.md` names the exact script for each table in its
Reproducibility section.

## Docs

`docs/FAILURE_ANALYSIS.md` is the full methodology writeup (grounding definition, flat vs.
conditional rate, the confound check). `docs/SCHEMA_CONFORMANCE.md` covers the second
failure axis. `docs/AUDIT_REPORT.md` documents two real bugs found and fixed mid-study
(both now covered by the regression tests in `tests/`). `docs/REGRESSION_TESTS.md`
explains what each of the 11 tests guards against.
