import pandas as pd
from typing import List, Tuple
import numpy as np


def split_log(log: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split the log chronologically into training and testing sets.
    """
    log = log.sort_values('start_timestamp')
    train_log = log.iloc[:int(len(log) * train_ratio)]
    test_log = log.iloc[int(len(log) * train_ratio):]
    return train_log, test_log

def split_log_train_val_test(log: pd.DataFrame, train_ratio: float = 0.64, val_ratio: float = 0.16, test_ratio: float = 0.2):
    """
    Split the log into training, validation, and testing sets.
    """
    log = log.sort_values('start_timestamp')
    train_log = log.iloc[:int(len(log) * train_ratio)]
    val_log = log.iloc[int(len(log) * train_ratio):int(len(log) * (train_ratio + val_ratio))]
    test_log = log.iloc[int(len(log) * (train_ratio + val_ratio)):]
    return train_log, val_log, test_log

def split_log_random(log: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split the log randomly into training and testing sets.
    """
    log = log.sample(frac=1).reset_index(drop=True)
    train_log = log.iloc[:int(len(log) * train_ratio)]
    test_log = log.iloc[int(len(log) * train_ratio):]
    return train_log, test_log


def get_unique_values_for_categorical_features(log: pd.DataFrame, categorical_features: List[str]):
    cat_features_dict = {}
    for cat_feature in categorical_features:
        cat_features_dict[cat_feature] = list(log[cat_feature].unique())
    return cat_features_dict



def build_decision_dataset(
    df: pd.DataFrame,
    feature_fn,
    negatives_per_pos: int = 5,
    random_state: int = 0
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Produces a pointwise dataset with per-decision groups.
    Returns (dataset_df, decision_meta_list).
    dataset_df columns: decision_id, agent, case_id, y, decision_time, x0..xd-1
    decision_meta includes full candidate sets for evaluation.
    """
    rng = np.random.default_rng(random_state)
    # Also seed global NumPy RNG for any implicit calls
    if random_state is not None:
        np.random.seed(random_state)

    # Resolve dynamic identifiers and timestamps
    case_id_col = 'case_id' if 'case_id' in df.columns else 'id'
    arrival_col = 'arrival' if 'arrival' in df.columns else ('enabled_time' if 'enabled_time' in df.columns else None)
    if arrival_col is None:
        raise ValueError("Missing required arrival column: expected 'arrival' or 'enabled_time'.")
    required = {case_id_col, 'agent', 'start_timestamp', arrival_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce temporal columns to datetimes for safety
    for col in ['start_timestamp', 'end_timestamp', 'arrival', 'enabled_time', 'due_date']:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Only rows where a start happened (decision moments)
    decisions = df.dropna(subset=['start_timestamp']).copy()
    decisions = decisions.sort_values('start_timestamp')

    # Universe of cases (attributes per case)
    # Include all columns to ensure feature_fn has everything it needs (dynamic features)
    attrib_cols = list(df.columns)
    # all_cases = df[attrib_cols].drop_duplicates(case_id_col)
    all_cases = df.copy()

    rows = []
    decision_meta = []
    decision_counter = 0

    for _, drow in decisions.iterrows():
        if pd.isna(drow['agent']) or drow['agent'] == '' or drow['agent'] == 'nan':
            continue
        agent = drow['agent']
        t = drow['start_timestamp']
        chosen_case = drow[case_id_col]
        candidates = all_cases[
            (all_cases[arrival_col] <= t) &
            (all_cases['start_timestamp'] >= t) 
        ]
        # check if mutliple cases with same id are in candidates, if so, remove cases in candidates that are not the first one based on start_timestamp
        candidates = candidates.sort_values('start_timestamp')
        candidates = candidates.drop_duplicates(case_id_col, keep='first')
        candidate_ids = candidates[case_id_col].tolist()
        if chosen_case not in candidate_ids:
            continue  # inconsistent; skip

        # Negative sampling for training rows (can be 0 for evaluation builds)
        neg_pool = [cid for cid in candidate_ids if cid != chosen_case]
        n_neg = min(negatives_per_pos, len(neg_pool))
        if negatives_per_pos > 0 and n_neg == 0:
            continue

        candidates_idx = candidates.set_index(case_id_col)

        # Positive
        pos_row = candidates_idx.loc[chosen_case]
        x_pos = feature_fn(agent, pos_row, t, candidates)
        rows.append({
            'decision_id': decision_counter,
            'agent': agent,
            'id': chosen_case,
            'y': 1,
            'decision_time': t,
            'features': x_pos
        })

        # Negatives
        if n_neg > 0:
            neg_samples = list(rng.choice(neg_pool, size=n_neg, replace=False))
            for cid in neg_samples:
                neg_row = candidates_idx.loc[cid]
                x_neg = feature_fn(agent, neg_row, t, candidates)
                rows.append({
                    'decision_id': decision_counter,
                    'agent': agent,
                    'id': cid,
                    'y': 0,
                    'decision_time': t,
                    'features': x_neg
                })

        # Meta for evaluation (always keep full candidate list)
        decision_meta.append({
            'decision_id': decision_counter,
            'agent': agent,
            'time': t,
            'chosen_id': chosen_case,
            'candidate_ids': candidate_ids
        })
        decision_counter += 1

    if not rows:
        raise ValueError("No samples constructed. Check log content and timestamps.")

    d = len(rows[0]['features'])
    feat_cols = [f'x{i}' for i in range(d)]
    out = []
    for r in rows:
        rec = {k: r[k] for k in ['decision_id', 'agent', 'id', 'y', 'decision_time']}
        rec.update({f'x{i}': r['features'][i] for i in range(d)})
        out.append(rec)
    dataset = pd.DataFrame(out)
    return dataset, decision_meta