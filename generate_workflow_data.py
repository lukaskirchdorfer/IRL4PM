import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import heapq

# Simulation parameters
START_TIME = datetime(2025, 1, 6, 9, 0)  # Monday, 9:00 AM
# N_CASES = 100
INTER_ARRIVAL_MEAN = 60  # in minutes
WORK_HOURS = (9, 17)  # 9am to 5pm
WORK_DAYS = set([0, 1, 2, 3, 4])  # Monday to Friday
PROCESSING_TIME_MEAN = 240  # mean processing time in minutes
DUE_DATE_MEAN = 800  # mean due date in minutes

CUSTOMER_LEVELS = ['platinum', 'gold', 'silver']
REGIONS = {
    'EMEA': 0.2,
    'US': 0.7,
    'Africa': 0.1
}

# Generate case arrival times
def generate_arrivals(n_cases):
    arrivals = []
    current_time = START_TIME
    for i in range(n_cases):
        while current_time.weekday() not in WORK_DAYS or not (WORK_HOURS[0] <= current_time.hour < WORK_HOURS[1]):
            current_time += timedelta(minutes=1)
        arrivals.append(current_time)
        inter_arrival = np.random.exponential(INTER_ARRIVAL_MEAN)
        current_time += timedelta(minutes=inter_arrival)
    return arrivals

# Define agent behavior
def agent_platinum_priority(queue):
    platinum_cases = [case for case in queue if case['level'] == 'platinum']
    return platinum_cases[0] if platinum_cases else random.choice(queue)

def agent_level_priority(queue):
    platinum_cases = [case for case in queue if case['level'] == 'platinum']
    gold_cases = [case for case in queue if case['level'] == 'gold']
    silver_cases = [case for case in queue if case['level'] == 'silver']
    level_cases = platinum_cases + gold_cases + silver_cases
    return level_cases[0]

def agent_region_specific(queue):
    region_cases = [case for case in queue if case['region'] in ['US']]
    return region_cases[0] if region_cases else None

def agent_high_value(queue):
    sorted_cases = sorted(queue, key=lambda c: -c['value'])
    return sorted_cases[0]

def agent_first_come_first_serve(queue):
    sorted_cases = sorted(queue, key=lambda c: c['arrival'])
    return sorted_cases[0]

def agent_random(queue):
    return random.choice(queue)

def agent_earliest_due_date(queue):
    sorted_cases = sorted(queue, key=lambda c: c['due_date'])
    return sorted_cases[0]

def agent_smallest_processing_time(queue):
    sorted_cases = sorted(queue, key=lambda c: c['processing_time'])
    return sorted_cases[0]



# Simulate employee behavior
class Employee:
    def __init__(self, name, selector):
        self.name = name
        self.selector = selector
        self.next_available = START_TIME

    def is_available(self, time):
        return self.next_available <= time

    def assign_case(self, queue, current_time):
        eligible = [case for case in queue if current_time >= case['arrival']]
        if not eligible:
            return None

        selected = self.selector(eligible)
        if selected is None:
            return None

        # processing_time = np.random.exponential(PROCESSING_TIME_MEAN)
        processing_time = selected['processing_time']
        start_time = max(current_time, self.next_available)
        end_time = start_time + timedelta(minutes=processing_time)
        self.next_available = end_time
        queue.remove(selected)
        return {
            **selected,
            'agent': self.name,
            'start_timestamp': start_time,
            'end_timestamp': end_time,
        }

# Create case generator
def create_case(case_id, arrival_time):
    return {
        'id': case_id,
        'arrival': arrival_time,
        'level': random.choice(CUSTOMER_LEVELS),
        'region': random.choices(list(REGIONS.keys()), weights=list(REGIONS.values()))[0],
        'value': random.uniform(10, 1000),
        'processing_time': np.random.exponential(PROCESSING_TIME_MEAN),
        'due_date': arrival_time + timedelta(minutes=np.random.exponential(DUE_DATE_MEAN))
    }

def is_working_time(t):
    return t.weekday() in WORK_DAYS and WORK_HOURS[0] <= t.hour < WORK_HOURS[1]

# Main simulation loop
def run_simulation(n_cases, agents, name_output_file = "event_log.csv", seed=None):
    """"
    n_cases: number of cases to generate
    agents: dictionary of agents and their functions
    """
    # Seed RNGs for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    arrivals = generate_arrivals(n_cases)
    queue = [create_case(i, t) for i, t in enumerate(arrivals)]

    employees = [Employee(name, selector) for name, selector in agents.items()]

    # employees = [
    #     Employee('Alice', agent_platinum_priority),
    #     Employee('Bob', agent_region_specific),
    #     Employee('Carol', agent_high_value)
    # ]

    time = START_TIME
    log = []

    while queue:
        if not is_working_time(time):
            time += timedelta(minutes=1)
            continue

        # randomly shuffle employees
        random.shuffle(employees)
        for employee in employees:
            if employee.is_available(time):
                result = employee.assign_case(queue, time)
                if result:
                    log.append(result)
        time += timedelta(minutes=1)

    df = pd.DataFrame(log)
    print(df.columns)
    df = df[['id', 'arrival', 'level', 'region', 'value', 'agent', 'start_timestamp', 'end_timestamp', 'processing_time', 'due_date']]
    df.to_csv(name_output_file, index=False)
    print(f"Simulation completed. Log saved to {name_output_file}.")

