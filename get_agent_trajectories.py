import pandas as pd
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
        }
        for feature in numerical_features:
            case[feature] = getattr(row, feature)
        for feature in categorical_features:
            case[feature] = getattr(row, feature)
        all_cases.append(case)

    # # Track all arrival events
    # all_cases = [{
    #     'id': row.id,
    #     'arrival': row.arrival,
    #     'level': row.level,
    #     'region': row.region,
    #     'value': row.value,
    #     'processing_time': row.processing_time,
    #     'due_date': row.due_date,
    # } for row in log.itertuples()]

    agents = log.agent.unique()

    # Extract trajectories for each agent
    trajectories = {agent: [] for agent in agents}

    # build once after sorting
    first_take = (log.sort_values('start_timestamp')
                    .drop_duplicates(case_id_feature, keep='first'))
    taken_time = dict(zip(first_take[case_id_feature], first_take['start_timestamp']))

    for row in log.itertuples():
        agent = row.agent
        case_id = getattr(row, case_id_feature)
        decision_time = row.start_timestamp

        # Construct the queue at decision time t with same-time exclusion (except chosen)
        queue = []
        for case in all_cases:
            if case['arrival'] <= decision_time:
                tt = taken_time.get(case['id'], None)
                if tt is None or tt > decision_time or case['id'] == case_id:
                    queue.append(case)

        state = [dict(c) for c in queue]
        for case in state:
            case['time_since_arrival'] = (decision_time - case['arrival']).total_seconds() / 60
            if 'due_date' in case:
                case['time_until_due_date'] = (case['due_date'] - decision_time).total_seconds() / 60

        # The action is the selected case
        action = case_id

        # Record the (state, action) pair
        trajectories[agent].append((state, action))

        # Mark the case as assigned
        assigned_cases.add(case_id)

    return trajectories