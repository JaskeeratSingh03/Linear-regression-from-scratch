"""Linear Regression — console demo.

Run with:
    python3 main.py

Pick a number from the menu:
    1. Train on synthetic data (y = 2.5x + 10 + noise)
    2. Train on data/sample.csv
    3. Show the learned equation
    4. Predict y for a value of x
    5. Plot cost vs iteration
    6. Plot data + learned line
    0. Exit

Options 3–6 require you to train first (option 1 or 2).
"""

from linear_regression import (
    LinearRegression,
    ascii_line_plot,
    ascii_scatter_with_line,
    load_csv,
    make_synthetic,
)


MENU = """
========================================
   Linear Regression — from scratch
========================================
  1. Generate synthetic data and train
  2. Load data/sample.csv and train
  3. Show learned equation
  4. Predict y for a value of x
  5. Plot cost vs iteration
  6. Plot data + regression line
  0. Exit
----------------------------------------
"""


def train_synthetic():
    """Generate noisy linear data and fit a line through it."""
    n_samples = 50
    noise = 10.0
    X, y = make_synthetic(n_samples=n_samples, noise=noise, seed=42)
    print(f"Generated {n_samples} synthetic samples (true line: y = 2.5*x + 10 + noise).")
    print("Training...")
    model = LinearRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    print(f"Done. Final cost: {model.cost_history[-1]:.4f}")
    return model, X, y


def train_csv(path="data/sample.csv"):
    """Load a CSV and fit a line through it."""
    try:
        X, y = load_csv(path)
    except FileNotFoundError:
        print(f"Could not find '{path}'. Run from the project root, or supply a full path.")
        return None, None, None
    print(f"Loaded {len(X)} samples from {path}.")
    print("Training...")
    model = LinearRegression(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    print(f"Done. Final cost: {model.cost_history[-1]:.4f}")
    return model, X, y


def need_trained(state):
    """Return True if a model has been trained, otherwise print a hint."""
    if state["model"] is None:
        print("Train a model first (option 1 or 2).")
        return False
    return True


def main():
    state = {"model": None, "X": None, "y": None}

    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            model, X, y = train_synthetic()
            state["model"], state["X"], state["y"] = model, X, y

        elif choice == "2":
            path = input("CSV path (press Enter for data/sample.csv): ").strip() or "data/sample.csv"
            model, X, y = train_csv(path)
            if model is not None:
                state["model"], state["X"], state["y"] = model, X, y

        elif choice == "3":
            if need_trained(state):
                print(f"Learned equation: {state['model'].equation()}")

        elif choice == "4":
            if need_trained(state):
                try:
                    x_val = float(input("Enter x: "))
                except ValueError:
                    print("Please enter a number.")
                    continue
                pred = state["model"].predict([[x_val]])[0]
                print(f"For x = {x_val}, predicted y = {pred:.4f}")

        elif choice == "5":
            if need_trained(state):
                print("Cost vs iteration:")
                ascii_line_plot(state["model"].cost_history, height=15, width=50)

        elif choice == "6":
            if need_trained(state):
                print("Data (`o`) and learned line (`*`):")
                ascii_scatter_with_line(state["X"], state["y"], state["model"])

        elif choice == "0":
            print("Bye!")
            break

        else:
            print("Unrecognized option. Try again.")


if __name__ == "__main__":
    main()
