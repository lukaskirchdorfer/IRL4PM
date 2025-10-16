import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

# ---------- helpers: resolve column names ----------
def _cols(df: pd.DataFrame):
    agent_col = "agent" if "agent" in df.columns else "resource"
    id_col    = "case_id" if "case_id" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        raise ValueError("Need an ID column named either 'case_id' or 'id'.")
    return agent_col, id_col

# ---------- helpers: value binning ----------
def _fit_bins(series: pd.Series, n_bins: int = 5) -> np.ndarray:
    vals = series.dropna().astype(float).values
    if len(vals) == 0:
        # trivial fallback (2 edges -> 1 bin)
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(vals, qs))
    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9
    return edges.astype(float)

def _apply_bins(x: float, edges: np.ndarray) -> int:
    if pd.isna(x):
        return -1  # unseen bin; will get only smoothing mass
    b = int(np.digitize([float(x)], edges, right=False)[0] - 1)  # 0..len(edges)-2 nominally
    return max(0, min(b, len(edges) - 2))

# ---------- frequentist model (per-agent) ----------
def train_freq_baseline_per_agent(
    train_log: pd.DataFrame,
    n_value_bins: int = 5,
    alpha: float = 1.0  # Laplace smoothing used at scoring time
) -> Tuple[Dict[Any, Dict[Tuple[str, str, int, int, int, int], int]], Dict[Any, Dict[str, np.ndarray]]]:
    """
    Returns:
      counts_by_agent: { agent: {(level, region, value_bin, proc_bin, t_since_bin, t_until_bin): count_chosen, ...} }
      bins_by_agent: { agent: { 'value': edges, 'processing_time': edges, 'time_since_arrival': edges, 'time_until_due_date': edges } }
    """
    agent_col, id_col = _cols(train_log)
    needed = {agent_col, id_col, 'start_timestamp', 'arrival', 'level', 'region', 'value', 'processing_time', 'due_date'}
    missing = needed - set(train_log.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    decisions = train_log.dropna(subset=['start_timestamp']).copy()

    counts_by_agent: Dict[Any, Dict[Tuple[str, str, int, int, int, int], int]] = {}
    bins_by_agent: Dict[Any, Dict[str, np.ndarray]] = {}

    for agent, g in decisions.groupby(agent_col):
        # compute decision-relative time features for the CHOSEN examples (train)
        start_ts = pd.to_datetime(g['start_timestamp'])
        arr = pd.to_datetime(g['arrival'])
        due = pd.to_datetime(g['due_date'])
        t_since = (start_ts - arr).dt.total_seconds() / 60.0
        t_until = (due - start_ts).dt.total_seconds() / 60.0

        bins = {
            'value': _fit_bins(g['value'], n_bins=n_value_bins),
            'processing_time': _fit_bins(g['processing_time'], n_bins=n_value_bins),
            'time_since_arrival': _fit_bins(t_since, n_bins=n_value_bins),
            'time_until_due_date': _fit_bins(t_until, n_bins=n_value_bins),
        }
        bins_by_agent[agent] = bins

        ctr: Dict[Tuple[str, str, int, int, int, int], int] = {}
        for _, row in g.iterrows():
            lvl  = str(row.get('level'))
            reg  = str(row.get('region'))
            vbin = _apply_bins(row.get('value'), bins['value'])
            pbin = _apply_bins(row.get('processing_time'), bins['processing_time'])
            # time-dependent for the chosen case
            st = pd.to_datetime(row.get('start_timestamp'))
            ar = pd.to_datetime(row.get('arrival'))
            du = pd.to_datetime(row.get('due_date'))
            tsince_val = (st - ar).total_seconds() / 60.0 if pd.notna(st) and pd.notna(ar) else np.nan
            tuntil_val = (du - st).total_seconds() / 60.0 if pd.notna(du) and pd.notna(st) else np.nan
            tsbin = _apply_bins(tsince_val, bins['time_since_arrival'])
            tubin = _apply_bins(tuntil_val, bins['time_until_due_date'])
            key = (lvl, reg, vbin, pbin, tsbin, tubin)
            ctr[key] = ctr.get(key, 0) + 1
        counts_by_agent[agent] = ctr

    return counts_by_agent, bins_by_agent

def score_candidates_freq(
    candidate_rows: pd.DataFrame,
    counts: Dict[Tuple[str, str, int, int, int, int], int],
    bins: Dict[str, np.ndarray],
    decision_time: pd.Timestamp,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Score per candidate = Laplace-smoothed frequency for (level, region, value_bin, proc_bin, t_since_bin, t_until_bin).
    """
    scores = []
    for _, r in candidate_rows.iterrows():
        lvl = str(r.get('level'))
        reg = str(r.get('region'))
        vbin = _apply_bins(r.get('value'), bins['value'])
        pbin = _apply_bins(r.get('processing_time'), bins['processing_time'])
        arr = pd.to_datetime(r.get('arrival'))
        due = pd.to_datetime(r.get('due_date'))
        tsince_val = (decision_time - arr).total_seconds() / 60.0 if pd.notna(arr) else np.nan
        tuntil_val = (due - decision_time).total_seconds() / 60.0 if pd.notna(due) else np.nan
        tsbin = _apply_bins(tsince_val, bins['time_since_arrival'])
        tubin = _apply_bins(tuntil_val, bins['time_until_due_date'])
        key = (lvl, reg, vbin, pbin, tsbin, tubin)
        c = counts.get(key, 0)
        scores.append(c + alpha)  # smoothing ensures unseen patterns still get >0
    return np.array(scores, dtype=float)

# ---------- build decisions (re-usable) ----------
def build_decision_meta(log_df: pd.DataFrame) -> Tuple[List[dict], pd.DataFrame]:
    """
    Build evaluation decisions and candidate universe.
    Returns:
      decisions: list of {agent, time, chosen_case_id, candidate_case_ids}
      all_cases_df: indexed by case id with columns [arrival, start_timestamp, end_timestamp, value, level, region, processing_time, due_date]
    """
    agent_col, id_col = _cols(log_df)
    attrib_cols = [id_col, "arrival", "start_timestamp", "end_timestamp", "value", "level", "region", "processing_time", "due_date"]

    all_cases = log_df[attrib_cols].drop_duplicates(id_col)
    # Ensure datetimes for safe comparisons
    for c in ['arrival', 'start_timestamp', 'end_timestamp', 'due_date']:
        if c in all_cases.columns:
            all_cases[c] = pd.to_datetime(all_cases[c], errors='coerce')

    decisions_df = log_df.dropna(subset=['start_timestamp']).copy().sort_values('start_timestamp')
    decisions = []
    ac_idx = all_cases.set_index(id_col)

    for _, d in decisions_df.iterrows():
        t = pd.to_datetime(d['start_timestamp'])
        chosen = d[id_col]
        agent = d[agent_col]

        # cand = all_cases[
        #     (all_cases['arrival'] <= t) &
        #     ((all_cases['start_timestamp'].isna()) | (all_cases['start_timestamp'] >= t))
        # ]
        cand = all_cases[
            (all_cases['arrival'] <= t) &
            (
                (all_cases['start_timestamp'].isna()) |
                (all_cases['start_timestamp'] > t) |
                (all_cases[id_col] == chosen)
            )
        ]
        cids = cand[id_col].tolist()
        if chosen not in cids or len(cids) == 0:
            continue

        decisions.append({
            'agent': agent,
            'time': t,
            'chosen_case_id': chosen,
            'candidate_case_ids': cids
        })

    return decisions, ac_idx

# ---------- evaluation (per-agent Top-k) ----------
def evaluate_freq_baseline_per_agent(
    test_log: pd.DataFrame,
    counts_by_agent: Dict[Any, Dict[Tuple[str, str, int, int, int, int], int]],
    bins_by_agent: Dict[Any, Dict[str, np.ndarray]],
    ks: List[int] = [1, 2, 3],
    alpha: float = 1.0
) -> Dict[Any, Dict[str, float]]:
    decisions, all_cases = build_decision_meta(test_log)

    per_agent_hits: Dict[Any, Dict[int, int]] = {}
    per_agent_total: Dict[Any, int] = {}

    for meta in decisions:
        agent = meta['agent']
        if agent not in counts_by_agent or agent not in bins_by_agent:
            continue
        counts = counts_by_agent[agent]
        bins  = bins_by_agent[agent]

        chosen = meta['chosen_case_id']
        cids   = meta['candidate_case_ids']
        cdf    = all_cases.loc[cids]

        scores = score_candidates_freq(cdf, counts, bins, meta['time'], alpha=alpha)
        ranked_idx  = np.argsort(scores)[::-1]
        ranked_cids = [cids[i] for i in ranked_idx]

        if agent not in per_agent_total:
            per_agent_total[agent] = 0
            per_agent_hits[agent] = {k: 0 for k in ks}

        per_agent_total[agent] += 1
        for k in ks:
            if chosen in ranked_cids[:k]:
                per_agent_hits[agent][k] += 1

    results: Dict[Any, Dict[str, float]] = {}
    for agent, total in per_agent_total.items():
        agent_res = {}
        for k in ks:
            agent_res[f"top{k}_accuracy"] = per_agent_hits[agent][k] / total if total > 0 else np.nan
        agent_res["accuracy"] = agent_res["top1_accuracy"]
        results[agent] = agent_res

    # Pretty print
    for agent, metrics in results.items():
        print(f"Agent: {agent}")
        for k in ks:
            print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
        print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results


# ---------- example usage ----------
# log = pd.read_csv(name_output_file, parse_dates=['arrival','start_timestamp','end_timestamp'])
# train_log, test_log = split_log_random(log)
# counts_by_agent, value_bins_by_agent = train_freq_baseline_per_agent(train_log, n_value_bins=5, alpha=1.0)
# freq_results = evaluate_freq_baseline_per_agent(test_log, counts_by_agent, value_bins_by_agent, ks=[1,2,3], alpha=1.0)
# print(freq_results)
