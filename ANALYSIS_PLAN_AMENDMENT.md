# Amendment to ANALYSIS_PLAN.md — family count cut from 4 to 3

Written 2026-08-16. `ANALYSIS_PLAN.md` is pre-registered and is not
edited — this file records what changed and why, and reruns the
detectability justification at the new grid size using the same
method (`detectability.py`, copied to `detectability_3agents.py` with
`n_agents`/`n_users` parametrized and pointed at `tables/detectability_3agents.csv`
/ `tables/detectability_3agents_summary.csv` so the original 4-agent
run stays intact as historical record).

## Why the family count changed

**The cut from 4 agent families to 3 was forced by upstream tool-call
support, not by anything observed in real run data.** Of the four
verified-on-paper AWQ families (`MODEL_CANDIDATES.md`), only two
passed the real GPU smoke test (`tables/smoke_test.csv`):

- Mistral-7B-Instruct-v0.3-AWQ: litellm's `completion()` call returns
  `None` (not an exception) for vLLM's `mistral` tool-call-parser
  response shape — a real litellm<->vLLM incompatibility, reproduced
  at two different context lengths.
- InternLM2.5-7b-chat-4bit: vLLM 0.27.1's engine crashes during CUDA
  graph capture on this repo's custom remote-modeling code — a real
  vLLM-version incompatibility, not a config/flag issue.

Three replacement candidates (gemma-2-9b-it, Phi-3.5-mini-instruct,
Yi-1.5-9B-Chat) were then checked and found to have **no
vLLM-recognized tool-call parser at all**, as of current vLLM
docs/source — disqualified regardless of AWQ checkpoint availability
(`MODEL_CANDIDATES_REPLACEMENTS.md`). Only granite-3.1-8b-instruct-AWQ
had both a real AWQ checkpoint and a real vLLM parser
(`--tool-call-parser granite`).

None of this reflects a finding about model capability, task
difficulty, or agent behavior — it is entirely a function of which
open-weight tool-calling AWQ checkpoints happen to have vLLM parser
support today. The operator's decision: 3 working families is
acceptable (4 was a design preference, not a statistical requirement —
detectability is driven by task/seed count, not agent count; 3 agents
gives 3 pairs instead of 6, which is fewer simultaneous contrasts and
an easier Holm correction, not a weaker one).

## New grid (pending Granite's real smoke-test confirmation)

3 agents x 3 user models x 20 tasks x 2 seeds = 360 runs, using
Qwen2.5-7B-Instruct-AWQ, Llama-3.1-8B-Instruct-AWQ,
granite-3.1-8b-instruct-AWQ (each as both agent and user model, same
design as the original 4x4, just N=3). This amendment's detectability
numbers assume this 3x3 grid; if Granite's real smoke test fails, the
grid is 2 families and this document does not apply — that scenario
is a stop-and-report condition, not something resolved here.

## Real minimum detectable delta at n_agents=3, n_users=3

Rerun via `detectability_3agents.py` (same method as the original:
plant a known rank-inversion interaction on synthetic data at a given
delta, run the real `analyze.py` pipeline, sweep delta until the
omnibus LRT + targeted-contrast decision rule from `ANALYSIS_PLAN.md`
reliably detects it). Full sweep at `tables/detectability_3agents.csv`,
summary at `tables/detectability_3agents_summary.csv`.

| n_agents | n_tasks | n_seeds | smallest detected delta |
|---|---|---|---|
| 3 | 20 | 2 | 0.15 |
| 3 | 25 | 2 | 0.15 |
| 3 | 30 | 2 | 0.10 |

For comparison, the original pre-registered number
(`tables/detectability.csv`, n_agents=4/n_users=4, untouched by this
amendment): 20 tasks x 2 seeds -> 0.12.

**Reading these numbers:** dropping to 3 agents at the same 20-task
grid makes the design less sensitive (0.12 -> 0.15) — fewer agent
pairs means fewer chances for a real interaction to also land as a
Holm-significant targeted contrast, which the primary-claim rule in
`ANALYSIS_PLAN.md` requires alongside the omnibus LRT. 25 tasks does
not recover this (still 0.15) — the extra 5 tasks were not enough to
offset the lost pair. **30 tasks does recover it and then some**:
0.10, better than the original 4-agent/20-task number. If the 360-run
grid (3x3x20x2) is used as-is, its minimum detectable delta is 0.15,
worse than the original 0.12. If the task count is widened to 30
(3x3x30x2 = 540 runs), the minimum detectable delta improves to 0.10.

This is a real tradeoff for the operator to weigh against the GPU-hour
cost of 540 vs 360 runs (see Decision 2's timing report, blocked as of
this writing on getting at least one normally-terminated run to
measure from — 0 of 4 sampled Qwen-AWQ episodes terminated normally,
see `PROGRESS.md` and commit `50267d1`).

## Status update, 2026-08-16: this design is STOPPED, not just amended

Granite's real smoke test subsequently failed (~2.5 tok/s throughput,
timed out, 0 tool calls — `DECISIONS_NEEDED.md` #0), so the 3-family
grid above never became real — confirmed working set dropped back to
2 families (Qwen2.5-7B-AWQ, Llama-3.1-8B-AWQ). Separately, and more
fundamentally: 0 of 4 sampled Qwen-AWQ episodes terminated normally
(`DECISIONS_NEEDED.md` #2's update) — everything runs into
`too_many_errors` or the step cap, meaning there is no clean-completion
score spread left to compute a rank inversion over regardless of how
many families are available. The operator's decision: stop the
rank-inversion design entirely and characterize the failure mode
instead (`FAILURE_ANALYSIS.md`). Everything above this point in the
document is preserved as the real, correct detectability analysis for
the 3-agent grid *if that grid had gone forward* — it did not.

## Detectability at n_agents=2, n_users=2 (for the record, in case a
   2-family design is revisited later)

Requested by the operator alongside the stop-rank-inversion decision,
"so we know what is measurable at all if we stay with 2 families" —
informational, not a proposal to resume the rank-inversion design.
Same method, `detectability_2agents.py`, GRIDS=[(20,2),(30,2),(40,2)]:

| n_agents | n_tasks | n_seeds | smallest detected delta |
|---|---|---|---|
| 2 | 20 | 2 | 0.12 |
| 2 | 30 | 2 | 0.08 |
| 2 | 40 | 2 | 0.06 |

**Noteworthy, not intuitive:** 2 agents at 20 tasks (0.12) is BETTER
than 3 agents at the same 20 tasks (0.15, see above) and ties the
original 4-agent number. This is not a fluke of the synthetic
generator — with only 1 agent pair (vs 3 pairs at n=3, 6 pairs at
n=4), there are far fewer simultaneous targeted contrasts to
Holm-correct across (1 vs 9 vs 36), so the one contrast that matters
survives correction more easily. Fewer agents costs comparison
breadth (you can only ever compare the 2 you have), not necessarily
detection sensitivity on the pair you do compare. This does not change
the fact that the rank-inversion design is stopped for the reason
above (no score spread to detect anything with, at any grid size) —
it's recorded here purely so the number exists if the question comes
up again later.
