import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# --- helper: build decision metadata (same as before) ---
def build_decision_meta(log_df: pd.DataFrame, train_log_agents: List[str]) -> Tuple[List[dict], pd.DataFrame]:
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

    for _, d in decisions_df.iterrows():
        if pd.isna(d['agent']) or d['agent'] == '' or d['agent'] == 'nan' or d['agent'] not in train_log_agents:
            continue
        t = d['start_timestamp']
        chosen = d[case_id_col]
        agent = d['agent']

        arrival_col = 'arrival' if 'arrival' in all_cases.columns else ('enabled_time' if 'enabled_time' in all_cases.columns else None)
        if arrival_col is None:
            raise ValueError("Missing required arrival column: expected 'arrival' or 'enabled_time'.")

        # candidates available at time t
        # cand = all_cases[
        #     (all_cases['arrival'] <= t) &
        #     ((all_cases['start_timestamp'].isna()) | (all_cases['start_timestamp'] >= t))
        # ]
        # cand = all_cases[
        #     (all_cases['arrival'] <= t) &
        #     ((all_cases['start_timestamp'].isna()) | (all_cases['start_timestamp'] > t))
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

# --- FIFO baseline (per agent), top-k evaluation ---
def evaluate_fifo_baseline_per_agent(
    test_log: pd.DataFrame,
    train_log: pd.DataFrame,
    ks: List[int] = [1, 2, 3],
    stable_tiebreak: bool = False
) -> Dict[Any, Dict[str, float]]:
    """
    Per-agent FIFO baseline: rank by earliest 'arrival' among the candidate set.
    If stable_tiebreak=True, ties are broken deterministically by case_id to make results reproducible.
    Returns:
      { agent: {"top1_accuracy": ..., "top2_accuracy": ..., "top3_accuracy": ..., "accuracy": ...}, ... }
    """
    train_log_agents = train_log['agent'].unique().tolist()
    decisions, all_cases = build_decision_meta(test_log, train_log_agents)

    per_agent_hits: Dict[Any, Dict[int, int]] = {}
    per_agent_total: Dict[Any, int] = {}

    arrival_col = 'arrival' if 'arrival' in all_cases.columns else ('enabled_time' if 'enabled_time' in all_cases.columns else None)
    if arrival_col is None:
        raise ValueError("Missing required arrival column: expected 'arrival' or 'enabled_time'.")

    # Normalize arrival column: remove timezone and ensure datetime dtype
    if is_datetime64tz_dtype(all_cases[arrival_col]):
        # Convert to UTC then drop timezone to make tz-naive
        all_cases[arrival_col] = pd.to_datetime(all_cases[arrival_col], utc=True, errors='coerce').dt.tz_convert(None)
    if not is_datetime64_any_dtype(all_cases[arrival_col]):
        all_cases[arrival_col] = pd.to_datetime(all_cases[arrival_col], errors='coerce')

    for meta in decisions:
        agent = meta['agent']
        chosen = meta['chosen_case_id']
        cids = meta['candidate_case_ids']
        if len(cids) == 0:
            continue

        # Get candidate rows and sort by earliest arrival (FIFO)
        cdf = all_cases.loc[cids].copy()

        # Tie-breaker: deterministic secondary sort key (case_id) for reproducibility
        if stable_tiebreak:
            # preserve original index name to reference later
            idx_name = cdf.index.name or "case_id"
            cdf = cdf.reset_index().sort_values([arrival_col, idx_name], ascending=[True, True])
            ranked_cids = cdf[idx_name].tolist()
        else:
            cdf = cdf.sort_values(arrival_col, ascending=True)
            ranked_cids = cdf.index.tolist()

        # Init counters for this agent
        if agent not in per_agent_total:
            per_agent_total[agent] = 0
            per_agent_hits[agent] = {k: 0 for k in ks}

        per_agent_total[agent] += 1
        for k in ks:
            if chosen in ranked_cids[:k]:
                per_agent_hits[agent][k] += 1
            # else:
            #     print(f"Agent: {agent} chose {chosen} but ranked_cids[:{k}] does not contain it")
            #     print(ranked_cids)

    # Build results per agent
    results: Dict[Any, Dict[str, float]] = {}
    for agent, total in per_agent_total.items():
        agent_res = {}
        for k in ks:
            agent_res[f"top{k}_accuracy"] = per_agent_hits[agent][k] / total if total > 0 else np.nan
        agent_res["accuracy"] = agent_res.get("top1_accuracy", np.nan)
        results[agent] = agent_res

    # Pretty print
    for agent, metrics in results.items():
        print(f"Agent: {agent}")
        for k in ks:
            print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
        print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results

# --- example usage ---
# fifo_results = evaluate_fifo_baseline_per_agent(test_log, ks=[1,2,3])
# print(fifo_results)
