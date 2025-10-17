import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from functools import partial
from typing import Any, Dict, List, Tuple

# =========================================================
# 0) Helpers: encoders for categorical features (fit on train)
# =========================================================

def build_encoders(train_log: pd.DataFrame, categorical_features: List[str]) -> Dict[str, Dict[Any, int]]:
    """
    Build integer encoders for categorical features from the training data.
    Unknown categories at test time map to -1.
    """
    encoders = {}
    for col in categorical_features:
        if col not in train_log.columns:
            raise ValueError(f"Required column '{col}' not found in log.")
        cats = list(pd.Series(train_log[col].dropna().unique()).astype(str))
        mapping = {c: i for i, c in enumerate(sorted(cats))}
        encoders[col] = mapping
    return encoders

def encode_category(val: Any, mapping: Dict[Any, int]) -> int:
    return mapping.get(str(val), -1)

# =========================================================
# 1) Dynamic feature function using YAML configuration
# =========================================================

def feature_dynamic(
    agent: Any,
    case_row: pd.Series,
    decision_time: pd.Timestamp,
    candidates_df: pd.DataFrame,
    encoders: Dict[str, Dict[Any, int]],
    numerical_features: List[str],
    categorical_features: List[str],
    scalers: Dict[str, Any] = None
) -> np.ndarray:
    """
    Construct features dynamically based on YAML configuration.
    Features are processed in order: numerical_features, then categorical_features.
    """
    features = []
    
    # Process numerical features
    for feature in numerical_features:
        if feature == 'time_since_arrival':
            # Calculate time since arrival
            arr = case_row.get('arrival', case_row.get('enabled_time'))
            if pd.notna(arr) and not isinstance(arr, pd.Timestamp):
                arr = pd.to_datetime(arr)
            t_since = (decision_time - arr).total_seconds() / 60 if pd.notna(arr) else 0.0
            features.append(t_since)
        elif feature == 'time_until_due_date':
            # Calculate time until due date
            due = case_row.get('due_date')
            if pd.notna(due) and not isinstance(due, pd.Timestamp):
                due = pd.to_datetime(due)
            t_until = (due - decision_time).total_seconds() / 60 if pd.notna(due) else 0.0
            features.append(t_until)
        else:
            # Regular numerical feature
            val = case_row.get(feature, np.nan)
            val = float(val) if pd.notna(val) else 0.0
            # Apply scaling if scaler is provided
            if scalers is not None and feature in scalers:
                val = float(scalers[feature].transform(np.array([[val]])).ravel()[0])
            features.append(val)
    
    # Process categorical features
    for feature in categorical_features:
        if feature in encoders:
            cat_val = encode_category(case_row.get(feature, None), encoders[feature])
            features.append(float(cat_val))
        else:
            # If no encoder found, use 0
            features.append(0.0)
    
    return np.array(features, dtype=float)

# =========================================================
# 2) Build decision dataset (pointwise)
# =========================================================

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
    all_cases = df[attrib_cols].drop_duplicates(case_id_col)

    rows = []
    decision_meta = []
    decision_counter = 0

    for _, drow in decisions.iterrows():
        if pd.isna(drow['agent']):
            continue
        agent = drow['agent']
        t = drow['start_timestamp']
        chosen_case = drow[case_id_col]

        # Candidates: arrived and not yet started by t
        # candidates = all_cases[
        #     (all_cases['arrival'] <= t) &
        #     ((all_cases['start_timestamp'].isna()) | (all_cases['start_timestamp'] >= t))
        # ]
        candidates = all_cases[
            (all_cases[arrival_col] <= t) &
            (
                (all_cases['start_timestamp'].isna()) |
                (all_cases['start_timestamp'] > t) |
                (all_cases[case_id_col] == chosen_case)
            )
        ]
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

# =========================================================
# 3) Train per-agent Decision Trees (using only value/level/region)
# =========================================================

def train_decision_trees_per_agent(
    train_log: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    max_depth: int = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    negatives_per_pos: int = 5,
    scalers: Dict[str, Any] = None
) -> Tuple[Dict[Any, DecisionTreeClassifier], Dict[str, Dict[Any, int]]]:
    """
    Returns (models_by_agent, encoders) so the same encoders are used at test time.
    """
    encoders = build_encoders(train_log, categorical_features)
    feat_fn = lambda agent, row, t, cdf: feature_dynamic(
        agent, row, t, cdf, encoders=encoders, 
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        scalers=scalers
    )

    # Print the feature names in the order used by the feature function
    feature_names = list(numerical_features) + list(categorical_features)
    print("Decision Tree training feature names (order):", feature_names)

    train_ds, _ = build_decision_dataset(
        train_log,
        feature_fn=feat_fn,
        negatives_per_pos=negatives_per_pos,
        random_state=random_state
    )
    feature_cols = [c for c in train_ds.columns if c.startswith('x')]
    models = {}

    for agent, g in train_ds.groupby('agent'):
        X = g[feature_cols].values
        y = g['y'].values
        if len(np.unique(y)) < 2:
            # not enough signal
            continue
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        clf.fit(X, y)
        models[agent] = clf

    return models, encoders

# =========================================================
# 3B) Train a single global Decision Tree (all agents combined)
# =========================================================

def train_decision_tree_global(
    train_log: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    max_depth: int = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    negatives_per_pos: int = 5,
    scalers: Dict[str, Any] = None
) -> Tuple[DecisionTreeClassifier, Dict[str, Dict[Any, int]]]:
    """
    Train a single DecisionTreeClassifier across all agents using dynamic features.
    Returns (model, encoders).
    """
    encoders = build_encoders(train_log, categorical_features)
    feat_fn = lambda agent, row, t, cdf: feature_dynamic(
        agent, row, t, cdf, encoders=encoders, 
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        scalers=scalers
    )

    # Print the feature names in the order used by the feature function
    feature_names = list(numerical_features) + list(categorical_features)
    print("Global Decision Tree training feature names (order):", feature_names)

    train_ds, _ = build_decision_dataset(
        train_log,
        feature_fn=feat_fn,
        negatives_per_pos=negatives_per_pos,
        random_state=random_state
    )

    feature_cols = [c for c in train_ds.columns if c.startswith('x')]
    X = train_ds[feature_cols].values
    y = train_ds['y'].values

    # Guard: need at least two classes to fit
    if len(np.unique(y)) < 2:
        raise ValueError("Not enough class diversity to train a global tree.")

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    clf.fit(X, y)
    return clf, encoders

# =========================================================
# 4) Evaluate top-k accuracy (1/2/3) using the same features
# =========================================================

import numpy as np
import pandas as pd
from typing import Any, Dict, List

def evaluate_topk_decisions_per_agent(
    test_log: pd.DataFrame,
    models: Dict[Any, "DecisionTreeClassifier"],
    encoders: Dict[str, Dict[Any, int]],
    numerical_features: List[str],
    categorical_features: List[str],
    ks: List[int] = [1, 2, 3],
    scalers: Dict[str, Any] = None
) -> Dict[Any, Dict[str, float]]:
    """
    Computes per-agent Top-k accuracies. Returns:
      { agent: {"top1_accuracy": ..., "top2_accuracy": ..., "top3_accuracy": ..., "accuracy": ...}, ... }
    Uses the same dynamic feature function as training.
    """

    # --- feature function with same encoders/scalers ---
    feat_fn = lambda agent, row, t, cdf: feature_dynamic(
        agent, row, t, cdf, encoders=encoders, 
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        scalers=scalers
    )

    # --- Build evaluation decisions (no negative subsampling; just meta) ---
    _, decisions = build_decision_dataset(
        test_log,
        feature_fn=feat_fn,
        negatives_per_pos=0,  # only need decision meta; features recomputed below
        random_state=0
    )

    # --- Resolve case id column name in test_log ---
    case_id_col = "case_id" if "case_id" in test_log.columns else "id"
    # Keep all columns to allow dynamic features
    all_cases = test_log.drop_duplicates(case_id_col).set_index(case_id_col)

    # --- Accumulators per agent ---
    per_agent_hits: Dict[Any, Dict[int, int]] = {}
    per_agent_total: Dict[Any, int] = {}

    for meta in decisions:
        agent = meta["agent"]
        if agent not in models:
            # skip agents without a trained model
            continue
        clf = models[agent]
        t = meta["time"]

        # meta keys: support both naming variants
        chosen = meta.get("chosen_case_id", meta.get("chosen_id"))
        cids = meta.get("candidate_case_ids", meta.get("candidate_ids"))

        # Some decisions might be empty or inconsistent; guard
        if chosen is None or cids is None or len(cids) == 0:
            continue
        if any(cid not in all_cases.index for cid in cids):
            # If any candidate is missing attributes, skip this decision
            continue

        cdf = all_cases.loc[cids]
        feats = [feat_fn(agent, row, t, cdf) for _, row in cdf.iterrows()]
        X = np.vstack(feats)

        proba = clf.predict_proba(X)
        one_idx = list(clf.classes_).index(1)
        scores = proba[:, one_idx]

        ranked_idx = np.argsort(scores)[::-1]
        ranked_cids = [cids[i] for i in ranked_idx]

        # init counters for this agent
        if agent not in per_agent_total:
            per_agent_total[agent] = 0
            per_agent_hits[agent] = {k: 0 for k in ks}

        per_agent_total[agent] += 1
        for k in ks:
            if chosen in ranked_cids[:k]:
                per_agent_hits[agent][k] += 1

    # --- Build results per agent ---
    results: Dict[Any, Dict[str, float]] = {}
    for agent, total in per_agent_total.items():
        agent_res = {}
        for k in ks:
            acc = per_agent_hits[agent][k] / total if total > 0 else np.nan
            agent_res[f"top{k}_accuracy"] = acc
        # convenience alias
        agent_res["accuracy"] = agent_res.get("top1_accuracy", np.nan)
        results[agent] = agent_res

    # # Pretty print
    # for agent, metrics in results.items():
    #     print(f"Agent: {agent}")
    #     for k in ks:
    #         print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
    #     print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results

# =========================================================
# 4B) Evaluate top-k accuracy (global single model)
# =========================================================

def evaluate_topk_decisions_global(
    test_log: pd.DataFrame,
    model: "DecisionTreeClassifier",
    encoders: Dict[str, Dict[Any, int]],
    numerical_features: List[str],
    categorical_features: List[str],
    ks: List[int] = [1, 2, 3],
    scalers: Dict[str, Any] = None
) -> Dict[Any, Dict[str, float]]:
    """
    Computes per-agent Top-k accuracies using a single global model.
    Returns: { agent: {"top1_accuracy": ..., "top2_accuracy": ..., "top3_accuracy": ..., "accuracy": ...}, ... }
    Uses the same dynamic feature function as training.
    """

    feat_fn = lambda agent, row, t, cdf: feature_dynamic(
        agent, row, t, cdf, encoders=encoders, 
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        scalers=scalers
    )

    # Build decision metadata without negatives
    _, decisions = build_decision_dataset(
        test_log,
        feature_fn=feat_fn,
        negatives_per_pos=0,
        random_state=0
    )

    # Resolve case id column name in test_log
    case_id_col = "case_id" if "case_id" in test_log.columns else "id"
    # Keep all columns to allow dynamic features
    all_cases = test_log.drop_duplicates(case_id_col).set_index(case_id_col)

    per_agent_hits: Dict[Any, Dict[int, int]] = {}
    per_agent_total: Dict[Any, int] = {}

    for meta in decisions:
        agent = meta["agent"]
        t = meta["time"]
        chosen = meta.get("chosen_case_id", meta.get("chosen_id"))
        cids = meta.get("candidate_case_ids", meta.get("candidate_ids"))

        if chosen is None or cids is None or len(cids) == 0:
            continue
        if any(cid not in all_cases.index for cid in cids):
            continue

        cdf = all_cases.loc[cids]
        feats = [feat_fn(agent, row, t, cdf) for _, row in cdf.iterrows()]
        X = np.vstack(feats)

        proba = model.predict_proba(X)
        one_idx = list(model.classes_).index(1)
        scores = proba[:, one_idx]

        ranked_idx = np.argsort(scores)[::-1]
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
            acc = per_agent_hits[agent][k] / total if total > 0 else np.nan
            agent_res[f"top{k}_accuracy"] = acc
        agent_res["accuracy"] = agent_res.get("top1_accuracy", np.nan)
        results[agent] = agent_res

    # # Pretty print per agent
    # for agent, metrics in results.items():
    #     print(f"Agent: {agent}")
    #     for k in ks:
    #         print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
    #     print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results
