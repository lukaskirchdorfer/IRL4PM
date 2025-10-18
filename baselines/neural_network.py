import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from functools import partial
from typing import Any, Dict, List, Tuple
from utils.utils import build_decision_dataset
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
            if 'arrival' in case_row.index:
                arr = case_row.get('arrival')
            else:
                arr = case_row.get('enabled_time')
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

# Legacy function for backward compatibility
def feature_value_level_region(
    agent: Any,
    case_row: pd.Series,
    decision_time: pd.Timestamp,
    candidates_df: pd.DataFrame,
    encoders: Dict[str, Dict[Any, int]],
    value_scaler=None
) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Construct features [value, level_code, region_code, processing_time, time_since_arrival, time_until_due_date].
    """
    # value (numeric)
    v = case_row.get('value', np.nan)
    v = float(v) if pd.notna(v) else 0.0
    if value_scaler is not None:
        v = float(value_scaler.transform(np.array([[v]])).ravel()[0])

    # level (categorical -> int code)
    lvl = encode_category(case_row.get('level', None), encoders['level'])

    # region (categorical -> int code)
    reg = encode_category(case_row.get('region', None), encoders['region'])

    # processing time
    proc = case_row.get('processing_time', np.nan)
    proc = float(proc) if pd.notna(proc) else 0.0

    # time-dependent features w.r.t. decision_time
    arr = case_row.get('arrival')
    due = case_row.get('due_date')
    if pd.notna(arr) and not isinstance(arr, pd.Timestamp):
        arr = pd.to_datetime(arr)
    if pd.notna(due) and not isinstance(due, pd.Timestamp):
        due = pd.to_datetime(due)
    t_since = (decision_time - arr).total_seconds() / 60 if pd.notna(arr) else 0.0
    t_until = (due - decision_time).total_seconds() / 60 if pd.notna(due) else 0.0

    return np.array([v, float(lvl), float(reg), proc, t_since, t_until], dtype=float)


# =========================================================
# 3) Train per-agent Neural Network models
# =========================================================

def train_neural_network_per_agent(
    train_log: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    hidden_layer_sizes: Tuple[int, ...] = (100, 50),
    activation: str = 'relu',
    solver: str = 'adam',
    alpha: float = 0.0001,
    learning_rate: str = 'constant',
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    random_state: int = 0,
    negatives_per_pos: int = 5,
    scalers: Dict[str, Any] = None,
    standardize_features: bool = True
) -> Tuple[Dict[Any, MLPClassifier], Dict[str, Dict[Any, int]], Dict[Any, StandardScaler]]:
    """
    Returns (models_by_agent, encoders, scalers_by_agent) so the same encoders/scalers are used at test time.
    """
    encoders = build_encoders(train_log, categorical_features)
    feat_fn = lambda agent, row, t, cdf: feature_dynamic(
        agent, row, t, cdf, encoders=encoders, 
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        scalers=scalers
    )

    train_ds, _ = build_decision_dataset(
        train_log,
        feature_fn=feat_fn,
        negatives_per_pos=negatives_per_pos,
        random_state=random_state
    )
    
    feature_cols = [c for c in train_ds.columns if c.startswith('x')]
    models = {}
    scalers = {}

    for agent, g in train_ds.groupby('agent'):
        X = g[feature_cols].values
        y = g['y'].values
        if len(np.unique(y)) < 2:
            # not enough signal
            continue
        
        # Standardize features for neural network
        scaler = None
        if standardize_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            # early_stopping=True,
            # validation_fraction=0.1,
            # n_iter_no_change=10
        )
        clf.fit(X, y)
        models[agent] = clf
        scalers[agent] = scaler

    return models, encoders, scalers

# =========================================================
# 4) Evaluate top-k accuracy (1/2/3) using the same features
# =========================================================

def evaluate_topk_decisions_per_agent(
    test_log: pd.DataFrame,
    models: Dict[Any, "MLPClassifier"],
    encoders: Dict[str, Dict[Any, int]],
    scalers: Dict[Any, StandardScaler],
    numerical_features: List[str],
    categorical_features: List[str],
    ks: List[int] = [1, 2, 3],
    feature_scalers: Dict[str, Any] = None
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
        scalers=feature_scalers
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
        scaler = scalers.get(agent, None)
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
        
        # Apply same standardization as training
        if scaler is not None:
            X = scaler.transform(X)

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

    # Pretty print
    for agent, metrics in results.items():
        print(f"Agent: {agent}")
        for k in ks:
            print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
        print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results
