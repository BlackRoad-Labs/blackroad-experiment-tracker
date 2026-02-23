# blackroad-experiment-tracker

![CI](https://github.com/BlackRoad-Labs/blackroad-experiment-tracker/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-proprietary-red)

ML experiment tracking and comparison for the BlackRoad OS platform.

## Features
- Track ML experiments with hyperparameters, metrics, and artifacts
- Per-step metric logging
- Side-by-side experiment comparison
- Best run selection (maximize or minimize any metric)
- Markdown report export
- SQLite-backed persistence

## Usage
```bash
# Create an experiment
python main.py create "ResNet Run 1" resnet50 --hyperparams '{"lr": 0.001, "epochs": 10}'

# Log metrics
python main.py log <id> accuracy 0.92 --step 1
python main.py log <id> loss 0.23 --step 1

# Finish
python main.py finish <id>

# Compare runs
python main.py compare <id1> <id2>

# Find best run
python main.py best accuracy

# Export report
python main.py report <id>
```

## Testing
```bash
python -m pytest test_experiment_tracker.py -v
```
