import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

# --- helper: build decision metadata (same logic as earlier) ---
def build_decision_meta(log_df: pd.DataFrame) -> Tuple[List[dict], pd.DataFrame]:
    """
    Build evaluation decisions and candidate universes.
    Returns:
      decisions: list of dicts with keys {agent, time, chosen_case_id, candidate_case_ids}
      all_cases_df: indexed by case_id with columns {attrib_cols}
    """
    case_id_col = "case_id" if "case_id" in log_df.columns else "id"
    attrib_cols = log_df.columns.tolist()

    all_cases = log_df[attrib_cols].drop_duplicates(case_id_col)
    decisions_df = log_df.dropna(subset=['start_timestamp']).copy().sort_values('start_timestamp')

    decisions = []
    ac_idx = all_cases.set_index(case_id_col)

    if 'arrival' in attrib_cols:
        arrival_col = 'arrival'
    elif 'enabled_time' in attrib_cols:
        arrival_col = 'enabled_time'
    else:
        raise ValueError("No arrival column found")

    for _, d in decisions_df.iterrows():
        t = d['start_timestamp']
        chosen = d[case_id_col]
        agent = d['agent']

        # candidates available at time t
        # cand = all_cases[
        #     (all_cases['arrival'] <= t) &
        #     ((all_cases['start_timestamp'].isna()) | (all_cases['start_timestamp'] >= t))
        # ]
        cand = all_cases[
            (all_cases[arrival_col] <= t) &
            (
                (all_cases['start_timestamp'].isna()) |
                (all_cases['start_timestamp'] > t) |
                (all_cases[case_id_col] == chosen)
            )
        ]
        cids = cand[case_id_col].tolist()
        if chosen not in cids or len(cids) == 0:
            continue

        decisions.append({
            'agent': agent,
            'time': t,
            'chosen_case_id': chosen,
            'candidate_case_ids': cids
        })

    return decisions, ac_idx

# --- random baseline (per agent), top-k evaluation ---
def evaluate_random_baseline_per_agent(
    test_log: pd.DataFrame,
    ks: List[int] = [1, 2, 3],
    n_runs: int = 1,
    random_state: int = 0,
) -> Dict[Any, Dict[str, float]]:
    """
    Per-agent random-choice baseline.
    For each decision, we generate a random ranking over the candidate set.
    If n_runs > 1, results are averaged over independent random rankings.

    Returns:
      { agent: {"top1_accuracy": ..., "top2_accuracy": ..., "top3_accuracy": ..., "accuracy": ...}, ... }
    """
    rng = np.random.default_rng(random_state)
    decisions, _ = build_decision_meta(test_log)

    # accumulators: sum over runs, then divide by totals
    per_agent_hits_sum: Dict[Any, Dict[int, float]] = {}
    per_agent_total: Dict[Any, int] = {}

    for _ in range(n_runs):
        # initialize per run
        per_agent_hits: Dict[Any, Dict[int, int]] = {}
        per_agent_cnt: Dict[Any, int] = {}

        for meta in decisions:
            agent = meta['agent']
            chosen = meta['chosen_case_id']
            cids = meta['candidate_case_ids']
            if len(cids) == 0:
                continue

            # random ranking
            perm = rng.permutation(len(cids))
            ranked_cids = [cids[i] for i in perm]

            # init
            if agent not in per_agent_cnt:
                per_agent_cnt[agent] = 0
                per_agent_hits[agent] = {k: 0 for k in ks}

            per_agent_cnt[agent] += 1
            for k in ks:
                if chosen in ranked_cids[:k]:
                    per_agent_hits[agent][k] += 1

        # accumulate into sums
        for agent, total in per_agent_cnt.items():
            per_agent_total[agent] = per_agent_total.get(agent, 0) + total
            if agent not in per_agent_hits_sum:
                per_agent_hits_sum[agent] = {k: 0.0 for k in ks}
            for k in ks:
                per_agent_hits_sum[agent][k] += per_agent_hits[agent][k]

    # average over runs and compute accuracies
    results: Dict[Any, Dict[str, float]] = {}
    for agent, total in per_agent_total.items():
        agent_res = {}
        for k in ks:
            # total decisions were counted n_runs times; divide by n_runs as well
            acc = per_agent_hits_sum[agent][k] / total
            agent_res[f"top{k}_accuracy"] = acc
        agent_res["accuracy"] = agent_res["top1_accuracy"]
        results[agent] = agent_res

    # pretty print
    for agent, metrics in results.items():
        print(f"Agent: {agent}")
        for k in ks:
            print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
        print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results

# --- example usage ---
# rand_results = evaluate_random_baseline_per_agent(test_log, ks=[1,2,3], n_runs=1000, random_state=42)
# print(rand_results)
