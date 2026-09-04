"""Read runs/, emit every final table and figure for the paper.

No hand-computed numbers anywhere - every reported value comes out of
this script. Primary metric is action-level correctness (read+write
tallies), not binary task success.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests

from schema import RunRecord

FAILED_STATUSES = ("error", "timeout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--tables-dir", type=Path, default=Path("tables"))
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args()


def family(model_name: str) -> str:
    """Family = last '-' separated token, e.g. 'agent0-qwen' -> 'qwen'."""
    return model_name.rsplit("-", 1)[-1]


def load_runs(runs_dir: Path) -> list[RunRecord]:
    paths = sorted(p for p in runs_dir.glob("*.json") if p.name != "ground_truth.json")
    if not paths:
        raise FileNotFoundError(f"no run files found under {runs_dir}")
    records = []
    for p in paths:
        try:
            records.append(RunRecord.load(p))
        except Exception:
            print(f"MALFORMED RUN FILE: {p}", file=sys.stderr)
            raise
    return records


def to_dataframe(records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        total = r.action_total
        correct = r.action_correct
        rows.append(
            {
                "run_id": r.run_id,
                "task_id": r.task_id,
                "agent_model": r.agent_model,
                "user_model": r.user_model,
                "seed": r.seed,
                "status": r.status,
                "termination_reason": r.termination_reason,
                "task_success": r.task_success,
                "reward": r.reward,
                "n_turns": r.n_turns,
                "read_count": r.read.count,
                "read_correct": r.read.correct,
                "write_count": r.write.count,
                "write_correct": r.write.correct,
                "action_total": total,
                "action_correct": correct,
                "action_accuracy": (correct / total) if total > 0 else np.nan,
                "wall_clock_seconds": r.wall_clock_seconds,
            }
        )
    return pd.DataFrame(rows)


def audit_grid(df: pd.DataFrame) -> None:
    agents = sorted(df.agent_model.unique())
    users = sorted(df.user_model.unique())
    tasks = sorted(df.task_id.unique())
    seeds = sorted(df.seed.unique())

    expected = set(itertools.product(tasks, agents, users, seeds))
    found = set(zip(df.task_id, df.agent_model, df.user_model, df.seed))
    missing = expected - found

    n_completed = (df.status == "completed").sum()
    n_failed = df.status.isin(FAILED_STATUSES).sum()

    print("=== grid audit ===")
    print(f"agents ({len(agents)}): {agents}")
    print(f"user models ({len(users)}): {users}")
    print(f"tasks: {len(tasks)}, seeds: {len(seeds)}")
    print(f"expected cells (task x agent x user x seed): {len(expected)}")
    print(f"found run files: {len(df)}")
    print(f"completed: {n_completed}")
    print(f"failed (error/timeout): {n_failed}")
    print(f"missing cells: {len(missing)}")
    if missing:
        print("MISSING CELLS (task, agent, user, seed), up to 20 shown:")
        for cell in list(sorted(missing))[:20]:
            print(f"  {cell}")
    if len(found) != len(df):
        dupes = df[df.duplicated(subset=["task_id", "agent_model", "user_model", "seed"], keep=False)]
        print(f"DUPLICATE RUN FILES for the same cell: {len(dupes)} rows")
        print(dupes[["run_id", "task_id", "agent_model", "user_model", "seed"]])


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def output_score_matrix(df: pd.DataFrame, tables_dir: Path, figures_dir: Path, rng: np.random.Generator, n_boot: int, alpha: float) -> pd.DataFrame:
    agents = sorted(df.agent_model.unique())
    users = sorted(df.user_model.unique())
    clean = df[(df.status == "completed") & df.action_accuracy.notna()]

    rows = []
    for agent in agents:
        for user in users:
            vals = clean[(clean.agent_model == agent) & (clean.user_model == user)].action_accuracy.values
            n = len(vals)
            if n == 0:
                print(f"EMPTY CELL: agent={agent} user={user} - no completed runs with action data")
                mean, lo, hi = np.nan, np.nan, np.nan
            else:
                mean = float(vals.mean())
                lo, hi = bootstrap_ci(vals, n_boot, rng, alpha)
            rows.append({"agent_model": agent, "user_model": user, "n": n, "mean": mean, "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "score_matrix.csv", index=False)

    mat = table.pivot(index="agent_model", columns="user_model", values="mean").reindex(index=agents, columns=users)
    fig, ax = plt.subplots(figsize=(1.6 * len(users) + 2, 1.2 * len(agents) + 2))
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(users)), users, rotation=45, ha="right")
    ax.set_yticks(range(len(agents)), agents)
    for i in range(len(agents)):
        for j in range(len(users)):
            v = mat.values[i, j]
            text = "NA" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="white")
    ax.set_title("action accuracy: agent x user model")
    fig.colorbar(im, ax=ax, label="mean action accuracy")
    fig.tight_layout()
    fig.savefig(figures_dir / "score_matrix.png", dpi=150)
    plt.close(fig)
    return table


def output_rankings(score_table: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    agents = sorted(score_table.agent_model.unique())
    users = sorted(score_table.user_model.unique())

    rank_mat = pd.DataFrame(index=agents, columns=users, dtype=float)
    for user in users:
        col = score_table[score_table.user_model == user].set_index("agent_model")["mean"].reindex(agents)
        if col.isna().any():
            print(f"WARNING: cannot fully rank agents under user_model={user}, missing means for: "
                  f"{list(col[col.isna()].index)}")
        rank_mat[user] = col.rank(ascending=False, method="min")

    rank_mat.to_csv(tables_dir / "rankings.csv")

    detail = score_table.copy()
    detail["rank"] = detail.groupby("user_model")["mean"].rank(ascending=False, method="min")
    detail = detail.sort_values(["user_model", "rank"])
    detail.to_csv(tables_dir / "rankings_detail.csv", index=False)

    fig, ax = plt.subplots(figsize=(1.8 * len(users) + 2, 1.2 * len(agents) + 2))
    x = range(len(users))
    for agent in agents:
        y = rank_mat.loc[agent, users].values.astype(float)
        ax.plot(x, y, marker="o", label=agent)
    ax.set_xticks(list(x), users, rotation=45, ha="right")
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("agent ranking by user model")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(figures_dir / "rankings_bump.png", dpi=150)
    plt.close(fig)
    return rank_mat


def output_kendall_tau(rank_mat: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    users = list(rank_mat.columns)
    usable = [u for u in users if rank_mat[u].notna().all()]
    if len(usable) < len(users):
        print(f"WARNING: dropping user models with incomplete rankings from Kendall tau: "
              f"{set(users) - set(usable)}")

    tau_mat = pd.DataFrame(index=usable, columns=usable, dtype=float)
    rows = []
    for u1, u2 in itertools.combinations(usable, 2):
        tau, pval = kendalltau(rank_mat[u1].values, rank_mat[u2].values)
        tau_mat.loc[u1, u2] = tau
        tau_mat.loc[u2, u1] = tau
        rows.append({"user_model_1": u1, "user_model_2": u2, "tau": tau, "pvalue": pval})
    for u in usable:
        tau_mat.loc[u, u] = 1.0

    pd.DataFrame(rows).to_csv(tables_dir / "kendall_tau_pairs.csv", index=False)
    tau_mat.to_csv(tables_dir / "kendall_tau_matrix.csv")

    fig, ax = plt.subplots(figsize=(1.6 * len(usable) + 2, 1.4 * len(usable) + 2))
    im = ax.imshow(tau_mat.values.astype(float), cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(usable)), usable, rotation=45, ha="right")
    ax.set_yticks(range(len(usable)), usable)
    for i in range(len(usable)):
        for j in range(len(usable)):
            v = tau_mat.values[i, j]
            ax.text(j, i, f"{float(v):.2f}", ha="center", va="center")
    ax.set_title("Kendall tau between agent rankings, by user model pair")
    fig.colorbar(im, ax=ax, label="tau")
    fig.tight_layout()
    fig.savefig(figures_dir / "kendall_tau.png", dpi=150)
    plt.close(fig)


def _cell_means(agent_idx: np.ndarray, user_idx: np.ndarray, acc: np.ndarray, n_agents: int, n_users: int) -> np.ndarray:
    sums = np.zeros((n_agents, n_users))
    counts = np.zeros((n_agents, n_users))
    np.add.at(sums, (agent_idx, user_idx), acc)
    np.add.at(counts, (agent_idx, user_idx), 1)
    if (counts == 0).any():
        raise RuntimeError("permutation produced an empty agent x user cell - sample size too small for this test")
    return sums / counts


def _mean_disagreement(means: np.ndarray) -> float:
    n_users = means.shape[1]
    taus = []
    for u1, u2 in itertools.combinations(range(n_users), 2):
        r1 = pd.Series(means[:, u1]).rank(ascending=False).values
        r2 = pd.Series(means[:, u2]).rank(ascending=False).values
        tau, _ = kendalltau(r1, r2)
        taus.append(tau)
    return 1.0 - float(np.mean(taus))


def output_rank_permutation_conservative(df: pd.DataFrame, tables_dir: Path, rng: np.random.Generator, n_perm: int) -> None:
    """Conservative rank-disagreement check, secondary only - underpowered by
    construction for small agent panels. See ANALYSIS_PLAN.md's "Rank permutation
    test" section for why; output_omnibus_lrt/output_targeted_contrasts are primary."""
    clean = df[(df.status == "completed") & df.action_accuracy.notna()].copy()
    agents = sorted(clean.agent_model.unique())
    users = sorted(clean.user_model.unique())
    if len(agents) < 2 or len(users) < 2:
        print("SKIPPING rank permutation test: need >=2 agents and >=2 user models")
        return

    agent_idx = clean.agent_model.map({a: i for i, a in enumerate(agents)}).values
    user_idx = clean.user_model.map({u: i for i, u in enumerate(users)}).values
    acc = clean.action_accuracy.values.astype(float)

    observed = _mean_disagreement(_cell_means(agent_idx, user_idx, acc, len(agents), len(users)))

    b = 0
    for _ in range(n_perm):
        permuted = rng.permutation(user_idx)
        stat = _mean_disagreement(_cell_means(agent_idx, permuted, acc, len(agents), len(users)))
        if stat >= observed:
            b += 1
    p_value = (b + 1) / (n_perm + 1)

    result = pd.DataFrame(
        [
            {
                "statistic": "1 - mean_kendall_tau across user-model-pair rankings",
                "observed": observed,
                "n_perm": n_perm,
                "b_ge_observed": b,
                "p_value": p_value,
                "note": "CONSERVATIVE CHECK ONLY - underpowered by construction for small agent panels, "
                        "discards effect magnitude. Not the primary inferential test.",
            }
        ]
    )
    result.to_csv(tables_dir / "rank_permutation_test_conservative.csv", index=False)

    print(f"[conservative/underpowered check] rank permutation test: observed disagreement={observed:.4f}, "
          f"p={p_value:.5f} ({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05) - "
          f"discrete rank statistic, do not treat as the primary claim")


def _build_action_df(clean: pd.DataFrame) -> pd.DataFrame:
    """Explode completed runs into one row per action (correct: 0/1)."""
    action_rows = []
    for r in clean.itertuples():
        for count, correct in ((r.read_count, r.read_correct), (r.write_count, r.write_correct)):
            wrong = count - correct
            if correct:
                action_rows.append(
                    pd.DataFrame(
                        {"task_id": r.task_id, "agent_model": r.agent_model, "user_model": r.user_model,
                         "correct": np.ones(correct, dtype=int)}
                    )
                )
            if wrong:
                action_rows.append(
                    pd.DataFrame(
                        {"task_id": r.task_id, "agent_model": r.agent_model, "user_model": r.user_model,
                         "correct": np.zeros(wrong, dtype=int)}
                    )
                )
    if not action_rows:
        return pd.DataFrame(columns=["task_id", "agent_model", "user_model", "correct"])
    return pd.concat(action_rows, ignore_index=True)


def _elbo(result, model) -> float:
    mean_vec = np.concatenate([result.fe_mean, result.vc_mean, result.vcp_mean])
    sd_vec = np.concatenate([result.fe_sd, result.vc_sd, result.vcp_sd])
    return float(model.vb_elbo(mean_vec, sd_vec))


def output_omnibus_lrt(df: pd.DataFrame, tables_dir: Path) -> None:
    """Primary inferential test: does the agent x user_model interaction improve
    fit (outcome ~ agent*user_model + (1|task) vs. without the interaction), via
    an approximate (variational) likelihood-ratio test. See ANALYSIS_PLAN.md's
    "Primary claim: rank inversion" section for the full decision rule."""
    clean = df[df.status == "completed"].copy()
    if clean.empty:
        print("SKIPPING omnibus LRT: no completed runs")
        return
    action_df = _build_action_df(clean)
    if action_df.empty:
        print("SKIPPING omnibus LRT: no individual actions recorded")
        return

    try:
        m_add = BinomialBayesMixedGLM.from_formula(
            "correct ~ C(agent_model) + C(user_model)", {"task": "0 + C(task_id)"}, data=action_df
        )
        r_add = m_add.fit_vb()
        m_int = BinomialBayesMixedGLM.from_formula(
            "correct ~ C(agent_model) * C(user_model)", {"task": "0 + C(task_id)"}, data=action_df
        )
        r_int = m_int.fit_vb()
    except Exception:
        print("OMNIBUS LRT FAILED TO FIT:")
        traceback.print_exc()
        pd.DataFrame([{"elbo_additive": np.nan, "elbo_interaction": np.nan, "df_diff": np.nan,
                        "lr_stat": np.nan, "p_value": np.nan, "note": "FIT_FAILED, see stdout traceback"}]).to_csv(
            tables_dir / "omnibus_lrt.csv", index=False
        )
        return

    ok = (np.all(np.isfinite(r_add.fe_mean)) and np.all(np.isfinite(r_int.fe_mean))
          and np.all(np.isfinite(r_add.vcp_mean)) and np.all(np.isfinite(r_int.vcp_mean)))
    if not ok:
        print("OMNIBUS LRT MODEL DID NOT CONVERGE: non-finite values in fit result")

    e_add = _elbo(r_add, m_add)
    e_int = _elbo(r_int, m_int)
    df_diff = len(m_int.exog_names) - len(m_add.exog_names)
    lr_stat = 2 * (e_int - e_add)
    p_value = float(chi2.sf(lr_stat, df_diff)) if lr_stat > 0 else 1.0

    pd.DataFrame(
        [
            {
                "elbo_additive": e_add,
                "elbo_interaction": e_int,
                "df_diff": df_diff,
                "lr_stat": lr_stat,
                "p_value": p_value,
                "converged": ok,
                "note": "APPROXIMATE LRT: statistic built from variational ELBO, not exact log-likelihood",
            }
        ]
    ).to_csv(tables_dir / "omnibus_lrt.csv", index=False)

    print(f"omnibus interaction LRT (VB-approximate): LR={lr_stat:.4f}, df={df_diff}, p={p_value:.5f} "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at alpha=0.05)")


def _sign_flip_test(d: np.ndarray, rng: np.random.Generator, max_exact_tasks: int = 20, n_sample: int = 200000) -> tuple[float, int]:
    """Exact two-sided sign-flip permutation test on paired differences d.

    Enumerates all 2^len(d) sign patterns exactly when feasible; falls back
    to random sampling above max_exact_tasks to bound memory.
    Returns (p_value, n_enumerated).
    """
    n = len(d)
    observed = abs(d.mean())
    if n <= max_exact_tasks:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1
        stats = np.abs(signs @ d) / n
        n_total = 2**n
    else:
        signs = rng.integers(0, 2, size=(n_sample, n)) * 2 - 1
        stats = np.abs(signs @ d) / n
        n_total = n_sample
    b = int((stats >= observed).sum())
    return (b + 1) / (n_total + 1), n_total


def output_targeted_contrasts(df: pd.DataFrame, tables_dir: Path, rng: np.random.Generator) -> None:
    """Per (agent pair, user-model pair) double-difference test: does the
    agent_i-vs-agent_j gap itself shift between user_a and user_b. Exact
    sign-flip permutation per contrast, Holm-corrected across all contrasts."""
    clean = df[(df.status == "completed") & df.action_accuracy.notna()].copy()
    agents = sorted(clean.agent_model.unique())
    users = sorted(clean.user_model.unique())
    if len(agents) < 2 or len(users) < 2:
        print("SKIPPING targeted contrasts: need >=2 agents and >=2 user models")
        return

    cell = clean.groupby(["agent_model", "user_model", "task_id"])["action_accuracy"].mean()

    rows = []
    for i, j in itertools.combinations(agents, 2):
        for a, b in itertools.combinations(users, 2):
            tasks = sorted(clean.task_id.unique())
            d_vals = []
            for t in tasks:
                try:
                    d = (cell[(i, a, t)] - cell[(j, a, t)]) - (cell[(i, b, t)] - cell[(j, b, t)])
                except KeyError:
                    continue
                d_vals.append(d)
            d_vals = np.array(d_vals, dtype=float)
            n_tasks = len(d_vals)
            if n_tasks < 3:
                print(f"WARNING: contrast ({i} vs {j}) x ({a} vs {b}) has only {n_tasks} tasks with all 4 cells present - skipping")
                rows.append({"agent_i": i, "agent_j": j, "user_a": a, "user_b": b, "n_tasks": n_tasks,
                             "observed_d_mean": np.nan, "raw_p": np.nan})
                continue
            p_val, n_enum = _sign_flip_test(d_vals, rng)
            rows.append({"agent_i": i, "agent_j": j, "user_a": a, "user_b": b, "n_tasks": n_tasks,
                         "observed_d_mean": float(d_vals.mean()), "raw_p": p_val})

    table = pd.DataFrame(rows)
    valid = table.raw_p.notna()
    if valid.any():
        _, holm_p, _, _ = multipletests(table.loc[valid, "raw_p"].values, method="holm")
        table.loc[valid, "holm_p"] = holm_p
    else:
        table["holm_p"] = np.nan
    table["significant"] = table.holm_p < 0.05
    table = table.sort_values("holm_p", na_position="last")
    table.to_csv(tables_dir / "targeted_contrasts.csv", index=False)

    n_sig = int(table.significant.sum())
    print(f"targeted contrasts: {len(table)} total, {n_sig} significant after Holm correction")
    if n_sig:
        print(table[table.significant][["agent_i", "agent_j", "user_a", "user_b", "observed_d_mean", "holm_p"]].to_string(index=False))


def output_mixed_effects(df: pd.DataFrame, tables_dir: Path) -> None:
    clean = df[df.status == "completed"].copy()
    if clean.empty:
        print("SKIPPING mixed-effects model: no completed runs")
        return

    action_df = _build_action_df(clean)
    if action_df.empty:
        print("SKIPPING mixed-effects model: no individual actions recorded")
        return
    action_df["family_match"] = (
        action_df.agent_model.map(family) == action_df.user_model.map(family)
    ).astype(int)

    try:
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ C(agent_model) + C(user_model) + family_match",
            {"task": "0 + C(task_id)"},
            data=action_df,
        )
        result = model.fit_vb()
    except Exception:
        print("MIXED-EFFECTS MODEL FAILED TO FIT:")
        traceback.print_exc()
        pd.DataFrame([{"term": "FIT_FAILED", "estimate": np.nan, "sd": np.nan, "note": "see stdout traceback"}]).to_csv(
            tables_dir / "mixed_effects.csv", index=False
        )
        return

    fe_mean, fe_sd = np.asarray(result.fe_mean), np.asarray(result.fe_sd)
    vcp_mean, vcp_sd = np.asarray(result.vcp_mean), np.asarray(result.vcp_sd)
    converged = bool(np.all(np.isfinite(fe_mean)) and np.all(np.isfinite(fe_sd))
                     and np.all(np.isfinite(vcp_mean)) and np.all(np.isfinite(vcp_sd)))
    if not converged:
        print("MIXED-EFFECTS MODEL DID NOT CONVERGE: non-finite values in fit result")

    rows = []
    for name, m, s in zip(model.exog_names, fe_mean, fe_sd):
        note = ""
        if name == "family_match":
            note = "SECONDARY: family-match cells are a small slice of the grid, treat as underpowered"
        rows.append({"term": name, "estimate": m, "sd": s, "converged": converged, "note": note})
    for name, m, s in zip(model.vcp_names, vcp_mean, vcp_sd):
        rows.append({"term": f"var({name}) [log-sd]", "estimate": m, "sd": s, "converged": converged, "note": "random effect variance component"})
    pd.DataFrame(rows).to_csv(tables_dir / "mixed_effects.csv", index=False)
    print(f"mixed-effects model converged={converged}, n_actions={len(action_df)}")


def output_termination_breakdown(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    counts = pd.crosstab(df.user_model, df.termination_reason)
    props = pd.crosstab(df.user_model, df.termination_reason, normalize="index")
    counts.to_csv(tables_dir / "termination_breakdown_counts.csv")
    props.to_csv(tables_dir / "termination_breakdown_proportions.csv")

    fig, ax = plt.subplots(figsize=(1.6 * len(counts) + 2, 5))
    bottom = np.zeros(len(props))
    for reason in props.columns:
        ax.bar(props.index, props[reason].values, bottom=bottom, label=reason)
        bottom += props[reason].values
    ax.set_ylabel("proportion of runs")
    ax.set_title("termination reason by user model")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "termination_breakdown.png", dpi=150)
    plt.close(fig)


def output_n_turns(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> None:
    stats = df.groupby("user_model")["n_turns"].agg(["mean", "median", "std", "count"]).reset_index()
    stats.to_csv(tables_dir / "n_turns_by_user.csv", index=False)

    fig, ax = plt.subplots(figsize=(1.6 * len(stats) + 2, 5))
    ax.bar(stats.user_model, stats["mean"], yerr=stats["std"].fillna(0), capsize=4)
    ax.set_ylabel("n_turns")
    ax.set_title("talkiness by user model (mean +/- sd)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "n_turns_by_user.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    records = load_runs(args.runs_dir)
    df = to_dataframe(records)
    audit_grid(df)

    steps = [
        ("score_matrix", lambda: output_score_matrix(df, args.tables_dir, args.figures_dir, rng, args.n_boot, args.alpha)),
    ]
    results: dict[str, object] = {}
    failed: list[str] = []

    print("\n=== running outputs ===")
    name, fn = steps[0]
    try:
        results["score_matrix"] = fn()
        print(f"[ok] {name}")
    except Exception:
        print(f"[FAILED] {name}")
        traceback.print_exc()
        failed.append(name)

    remaining = [
        ("rankings", lambda: output_rankings(results["score_matrix"], args.tables_dir, args.figures_dir)),
        ("kendall_tau", lambda: output_kendall_tau(results["rankings"], args.tables_dir, args.figures_dir)),
        ("omnibus_lrt", lambda: output_omnibus_lrt(df, args.tables_dir)),
        ("targeted_contrasts", lambda: output_targeted_contrasts(df, args.tables_dir, rng)),
        ("rank_permutation_conservative", lambda: output_rank_permutation_conservative(df, args.tables_dir, rng, args.n_perm)),
        ("mixed_effects", lambda: output_mixed_effects(df, args.tables_dir)),
        ("termination_breakdown", lambda: output_termination_breakdown(df, args.tables_dir, args.figures_dir)),
        ("n_turns", lambda: output_n_turns(df, args.tables_dir, args.figures_dir)),
    ]
    for name, fn in remaining:
        if name == "rankings" and "score_matrix" not in results:
            print(f"[SKIPPED] {name}: score_matrix did not produce output")
            failed.append(name)
            continue
        if name == "kendall_tau" and "rankings" not in results:
            print(f"[SKIPPED] {name}: rankings did not produce output")
            failed.append(name)
            continue
        try:
            out = fn()
            if out is not None:
                results[name] = out
            print(f"[ok] {name}")
        except Exception:
            print(f"[FAILED] {name}")
            traceback.print_exc()
            failed.append(name)

    print("\n=== summary ===")
    print(f"{len(remaining) + 1 - len(failed)}/{len(remaining) + 1} outputs succeeded")
    if failed:
        print(f"FAILED OUTPUTS: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
