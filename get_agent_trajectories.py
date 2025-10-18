import pandas as pd
import numpy as np
from typing import List
import json

def get_agent_trajectories(log: pd.DataFrame, numerical_features: List[str], categorical_features: List[str]):
    """
    Create agent trajectories from log as state-action pairs over time.
    Args:
        log: pd.DataFrame 
        numerical_features: List[str] of numerical features to use for learning
        categorical_features: List[str] of categorical features to use for learning
    Returns:
        trajectories: Dict[str, List[Tuple[List[dict], Any]]] of agent trajectories
    """
    # Coerce temporal columns to datetimes for safety
    for col in ['start_timestamp', 'end_timestamp', 'arrival', 'enabled_time', 'due_date']:
        if col in log.columns and not np.issubdtype(log[col].dtype, np.datetime64):
            log[col] = pd.to_datetime(log[col], errors='coerce')
    # Sort events by start time (when the agent made the decision)
    log = log.sort_values('start_timestamp').reset_index(drop=True)

    numerical_features = numerical_features.copy()
    categorical_features = categorical_features.copy()

    # get arrival feature
    if 'arrival' in log.columns:
        arrival_feature = 'arrival'
        numerical_features.remove('time_since_arrival')
    elif 'enabled_time' in log.columns:
        arrival_feature = 'enabled_time'
        numerical_features.remove('time_since_arrival')

    if 'due_date' in log.columns:
        numerical_features.remove('time_until_due_date')
        numerical_features.append('due_date')
    if 'case_id' in log.columns:
        case_id_feature = 'case_id'
    else:
        case_id_feature = 'id'

    # Track cases assigned so far
    assigned_cases = set()

    # build all_cases
    all_cases = []
    for row in log.itertuples():
        case = {
            'id': getattr(row, case_id_feature),
            'arrival': getattr(row, arrival_feature),
            'start_timestamp': getattr(row, 'start_timestamp'),
        }
        for feature in numerical_features:
            case[feature] = getattr(row, feature)
        for feature in categorical_features:
            case[feature] = getattr(row, feature)
        all_cases.append(case)

    trajectories = {}
    for row in log.itertuples():
        if pd.isna(row.agent) or row.agent == '' or row.agent == 'nan' or row.agent == 'Nan':
            continue
        agent = row.agent
        case_id = getattr(row, case_id_feature)
        decision_time = row.start_timestamp

        queue = []
        for case in all_cases:
            if case['arrival'] <= decision_time:
                if case['start_timestamp'] >= decision_time:
                    if case['id'] in [case['id'] for case in queue]:
                        if case['start_timestamp'] < [case['start_timestamp'] for case in queue if case['id'] == case['id']][0]:
                            queue.remove([case for case in queue if case['id'] == case['id']][0])
                            queue.append(case)
                    else:
                        queue.append(case)

        state = [dict(c) for c in queue]
        for case in state:
            case['time_since_arrival'] = (decision_time - case['arrival']).total_seconds() / 60
            if 'due_date' in case:
                case['time_until_due_date'] = (case['due_date'] - decision_time).total_seconds() / 60

        if state is None or len(state) == 0:
            print(f"state is None or len(state) == 0 for agent {agent} and case {case_id}")

        # The action is the selected case
        action = case_id

        # Record the (state, action) pair
        if agent not in trajectories:
            trajectories[agent] = []
        trajectories[agent].append((state, action))

        # Mark the case as assigned
        assigned_cases.add(case_id)

    return trajectories