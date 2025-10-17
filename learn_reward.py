import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from scipy.special import logsumexp

def case_to_features(case, cat_features_dict, numerical_features):
    onehot_features = []
    for cat_feature in cat_features_dict:
        onehot = [int(case[cat_feature] == val) for val in cat_features_dict[cat_feature]]
        onehot_features.extend(onehot)  # Use extend instead of append to flatten

    feature_vec = np.array([case[num_feature] for num_feature in numerical_features] + onehot_features)
    feature_names = numerical_features + [f"{val}" for cat_feature in cat_features_dict for val in cat_features_dict[cat_feature]]

    return feature_vec, feature_names

# --- Load and preprocess trajectories ---
def load_trajectories(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Convert trajectories to feature-space
def prepare_data(trajectories, cat_features_dict, numerical_features, scaler=None):
    if scaler is None:
        all_feature_vectors = []
        for state, _ in trajectories:
            for case in state:
                features, feature_names = case_to_features(case, cat_features_dict, numerical_features)
                all_feature_vectors.append(features)
        scaler = StandardScaler()
        scaler.fit(all_feature_vectors)

    processed_trajectories = []
    for state, action_id in trajectories:
        if len(state) == 0:
            continue
        state_features = []
        action_idx = None
        for i, case in enumerate(state):
            features, feature_names = case_to_features(case, cat_features_dict, numerical_features)
            scaled = scaler.transform([features])[0]
            state_features.append((case['id'], scaled))
            if case['id'] == action_id:
                action_idx = i
        processed_trajectories.append((state_features, action_idx))
    feature_dim = len(features)
    return processed_trajectories, scaler, feature_dim, feature_names

# --- MaxEnt IRL core ---
def maxent_irl(trajectories, feature_dim, lr=0.005, epochs=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(feature_dim)

    for epoch in range(epochs):
        grad = np.zeros_like(w)
        empirical_feat_exp = np.zeros_like(w)

        # Compute empirical feature expectations
        for state, action_idx in trajectories:
            _, features = state[action_idx]
            empirical_feat_exp += features

        # Compute expected feature expectations under current policy
        for state, _ in trajectories:
            exp_rewards = []
            features_list = []
            for _, features in state:
                r = np.dot(w, features)
                exp_rewards.append(np.exp(r))
                features_list.append(features)
            Z = np.sum(exp_rewards) + 1e-10
            probs = np.array(exp_rewards) / Z
            for prob, feat in zip(probs, features_list):
                grad += prob * feat

        grad = empirical_feat_exp - grad
        w += lr * grad

        #### just for tracking
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Gradient norm: {np.linalg.norm(grad):.4f}")
            log_likelihood = 0
            for state, action_idx in trajectories:
                rewards = [np.dot(w, f) for _, f in state]
                log_probs = rewards - logsumexp(rewards)
                log_likelihood += log_probs[action_idx]
            print("Log-likelihood:", log_likelihood)
        ####

    return w

# --- Structured Max-Margin IRL ---
def maxmargin_irl(trajectories, feature_dim, lr=0.001, epochs=100, margin=1.0, seed=None):
    """
    Learn reward weights using a structured max-margin loss.
    """
    if seed is not None:
        np.random.seed(seed)
    w = np.random.randn(feature_dim)

    for epoch in range(epochs):
        grad = np.zeros_like(w)
        num_violations = 0

        for state, action_idx in trajectories:
            chosen_id, chosen_feat = state[action_idx]

            for i, (alt_id, alt_feat) in enumerate(state):
                if i == action_idx:
                    continue

                score_expert = np.dot(w, chosen_feat)
                score_other = np.dot(w, alt_feat)

                if score_expert < score_other + margin:
                    # Hinge loss is violated → update
                    grad += chosen_feat - alt_feat
                    num_violations += 1

        w += lr * grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Violations: {num_violations}, Gradient norm: {np.linalg.norm(grad):.4f}")

    return w

def learn_reward(trajectories, cat_features_dict, numerical_features, seed=None, lr=0.005, epochs=100):
    # --- Run for each agent ---
    agents = list(trajectories.keys())
    weights = {}
    scalers = {}

    for agent in agents:
        print(f"\nTraining MaxEnt IRL for {agent.capitalize()}")
        # trajs = load_trajectories(f"{agent}_trajectory.json")
        trajs, scaler, feature_dim, feature_names = prepare_data(trajectories[agent], cat_features_dict=cat_features_dict, numerical_features=numerical_features, scaler=None)
        w = maxent_irl(trajs, feature_dim=feature_dim, lr=lr, epochs=epochs, seed=seed)
        # w = maxmargin_irl(trajs, feature_dim=feature_dim)
        weights[agent] = w
        scalers[agent] = scaler
        print(f"Learned weights for {agent}: {w}")

    # pretty print
    print("\nFeature names: ", feature_names)
    for agent, w in weights.items():
        print(f"{agent.capitalize()}: {w.round(3)}")
    
    return weights, scalers


def test_accuracy_of_case_selection(trajs, weights, greedy=False, tau=1.0):
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0

    for state, action_idx in trajs:
        # Extract feature vectors for candidate cases
        phi_list = [features for _, features in state]
        scores = [float(weights @ phi) for phi in phi_list]

        if greedy:
            ranked_actions = np.argsort(scores)[::-1]  # descending order
        else:
            # Softmax sampling (used only for prediction, ranking still from scores)
            probs = np.exp(np.array(scores) / tau)
            probs /= probs.sum()
            ranked_actions = np.argsort(scores)[::-1]

        # Top-1 accuracy
        if action_idx == ranked_actions[0]:
            top1_correct += 1
        # Top-2 accuracy
        if action_idx in ranked_actions[:2]:
            top2_correct += 1
        # Top-3 accuracy
        if action_idx in ranked_actions[:3]:
            top3_correct += 1

    n = len(trajs)
    print(f"Top-1 Accuracy: {top1_correct / n:.4f}")
    print(f"Top-2 Accuracy: {top2_correct / n:.4f}")
    print(f"Top-3 Accuracy: {top3_correct / n:.4f}")




def test_reward(test_trajectories, weights, scaler, cat_features_dict, numerical_features, ks=[1, 2, 3], greedy=True, tau=1.0):
    """
    Evaluate per-agent Top-k accuracies using learned IRL weights.

    Returns a dict in the same format as decision_tree evaluation:
      { agent: {"top1_accuracy": ..., "top2_accuracy": ..., "top3_accuracy": ..., "accuracy": ...}, ... }
    """
    results = {}

    for agent in weights.keys():
        if agent not in test_trajectories.keys():
            continue
        print(f"\nTesting MaxEnt IRL for {agent.capitalize()}")
        trajs, _, _, _ = prepare_data(test_trajectories[agent], cat_features_dict, numerical_features, scaler[agent])
        w = weights.get(agent)
        if w is None:
            continue

        total = 0
        hits = {k: 0 for k in ks}

        for state, action_idx in trajs:
            # Extract feature vectors for candidate cases
            phi_list = [features for _, features in state]
            scores = [float(w @ phi) for phi in phi_list]

            if greedy:
                ranked_actions = np.argsort(scores)[::-1]  # descending order
            else:
                # Softmax sampling (used only for prediction, ranking still from scores)
                probs = np.exp(np.array(scores) / tau)
                probs /= probs.sum()
                ranked_actions = np.argsort(scores)[::-1]

            total += 1
            for k in ks:
                if action_idx in ranked_actions[:k]:
                    hits[k] += 1

        metrics = {}
        for k in ks:
            metrics[f"top{k}_accuracy"] = (hits[k] / total) if total > 0 else np.nan
        metrics["accuracy"] = metrics.get("top1_accuracy", np.nan)

        # Store
        results[agent] = metrics

    # Pretty print to mirror decision_tree evaluator
    for agent, metrics in results.items():
        print(f"Agent: {agent}")
        for k in ks:
            print(f"  Top-{k} Accuracy: {metrics[f'top{k}_accuracy']:.4f}")
        print(f"  Accuracy (Top-1): {metrics['accuracy']:.4f}")

    return results