"""Synthetic data generator and CSV loader — stdlib only."""

import csv
import random


def make_synthetic(n_samples=50, noise=10.0, true_w=2.5, true_b=10.0, seed=42):
    """Generate noisy linear data: y = true_w * x + true_b + noise * gauss(0, 1).

    Returns (X, y) where:
        X is a list of single-feature rows: [[x1], [x2], ...]
        y is a list of floats: [y1, y2, ...]
    The seed makes the demo reproducible.
    """
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n_samples):
        x = rng.uniform(0, 100)
        y_val = true_w * x + true_b + noise * rng.gauss(0, 1)
        X.append([x])
        y.append(y_val)
    return X, y


def load_csv(path):
    """Load a CSV with a header row.

    Convention: the LAST column is the target (y), every other column is a feature.
    For the 1-feature case the CSV looks like:
        x,y
        1.0,12.8
        2.5,15.2
        ...

    Returns (X, y).
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"CSV '{path}' needs a header row plus at least one data row.")

    header = rows[0]
    data = rows[1:]
    n_features = len(header) - 1  # last column is the target

    X = []
    y = []
    for row in data:
        if not row or all(cell.strip() == "" for cell in row):
            continue  # skip blank lines
        features = [float(cell) for cell in row[:n_features]]
        target = float(row[n_features])
        X.append(features)
        y.append(target)
    return X, y
