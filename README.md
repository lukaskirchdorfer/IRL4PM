## IRL4PM: Learning Agent Behavior from Event Logs via Inverse Reinforcement Learning

This repository contains code to learn agent decision-making policies from event logs using Inverse Reinforcement Learning (IRL). It also includes several supervised and heuristic baselines for comparison.

### Environment setup (Conda + pip)

The project uses Python packages pinned in `requirements.txt`. The recommended setup is a fresh Conda environment and installing packages via pip inside that env.

```bash
# 1) Create and activate a new environment (Python 3.10 is recommended)
conda create -n irl4pm python=3.10 
conda activate irl4pm

# 2) Optional: ensure pip is available/updated in the env
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt
```

Notes:
- Some dependencies (e.g., `cvxopt`, `prophet`, `pm4py`, `tensorflow`, `torch`, `xgboost`) may require system toolchains. If installation fails on your platform, install those with Conda first where possible, then run `pip install -r requirements.txt` again to resolve the rest.

### Repository structure

```text
IRL4PM/
  baselines/                  # Supervised and heuristic baselines
  data_configs/               # Feature configuration per dataset
  input_data/                 # Example processed logs
  results/                    # Outputs and analysis
  utils/                      # Data split and helpers
  generate_workflow_data.py   # Synthetic data generation (optional)
  get_agent_trajectories.py   # Builds (state, action) trajectories per agent
  learn_reward.py             # MaxEnt IRL and evaluation utilities
  main.py                     # Entry point to run IRL and baselines
  requirements.txt
  README.md
```

### How to run

The main entry point is `main.py`. It supports IRL and multiple baselines. Default values are sensible; pass flags to override.

Basic usage (IRL on an example ticket log):

```bash
python main.py \
  --input-file input_data/ticket_log_FCFS.csv \
  --method irl \
  --data-config-file ticket_log.yaml \
  --greedy \
  --irl-lr 0.005 \
  --irl-epochs 100 \
```

Available methods for `--method`:
- `irl` (Inverse Reinforcement Learning)
- `dt_local` (Decision Trees per agent)
- `lr_local` (Logistic Regression per agent)
- `svm`
- `xgboost`
- `neural_network`
- `fifo` (First-In-First-Out heuristic)
- `random`

Common flags:
- `--seed` random seed (default: 42)
- `--ks` comma-separated Top-k values for evaluation (default: `1,2,3`)
- `--data-config-file` selects feature config from `data_configs/` (e.g., `ticket_log.yaml`, `BPI13.yaml`)

Outputs:
- A summary row is appended to `results/overall_results_BPI13.csv` for each run, including per-agent Top-k metrics and relevant hyperparameters.

### Data expectations

Input CSVs are expected in `input_data/` and must contain temporal columns specified in the selected data config (see `data_configs/*.yaml`). The script parses those columns as datetimes and builds agent trajectories accordingly.


