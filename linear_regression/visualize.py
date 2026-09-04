"""ASCII plots using only the standard library.

Two functions:
    ascii_line_plot(values, height, width)        — for the cost curve.
    ascii_scatter_with_line(X, y, model, ...)     — data + learned line.

Both intentionally simple — no external plotting libraries.
"""


def ascii_line_plot(values, height=15, width=50):
    """Plot a list of numbers as an ASCII line chart.

    Resamples `values` down to `width` columns and maps each value to a row.
    """
    if not values:
        print("(no data)")
        return

    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0  # avoid divide-by-zero on a flat series

    # Down-sample: take every Nth value so we land near `width` columns.
    step = max(1, len(values) // width)
    cols = values[::step][:width]

    # For each column, the row where the marker sits (0 = top).
    rows = [int((1 - (v - lo) / span) * (height - 1)) for v in cols]

    for r in range(height):
        line = ""
        for row in rows:
            line += "*" if row == r else " "
        print("|" + line)
    print("+" + "-" * len(cols))
    print(f"  cost: {lo:.2f} -> {hi:.2f}   ({len(values)} iterations)")


def ascii_scatter_with_line(X, y, model, height=20, width=60):
    """Show data points (`o`) and the model's predicted line (`*`) together.

    `X` is list-of-lists (we use only the first feature, so single-feature models).
    `y` is a flat list of targets.
    `model` must already be trained (call fit() first).
    """
    if not X or not y:
        print("(no data)")
        return

    coeffs = model.coefficients_original_scale()
    if coeffs is None:
        print("(plot only supports single-feature models)")
        return
    w, b = coeffs

    # Unpack the single feature.
    xs = [row[0] for row in X]
    ys = list(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # The line should span the data's x range.
    y_at_xmin = w * x_min + b
    y_at_xmax = w * x_max + b
    y_min = min(y_min, y_at_xmin, y_at_xmax)
    y_max = max(y_max, y_at_xmin, y_at_xmax)

    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    def to_col(x):
        return int((x - x_min) / x_span * (width - 1))

    def to_row(y_val):
        # Flip: row 0 = top, but higher y should be higher visually too.
        return int((1 - (y_val - y_min) / y_span) * (height - 1))

    # Build a blank grid. Each cell holds ' ' or a marker.
    grid = [[" "] * width for _ in range(height)]

    # Paint the regression line first (so points overwrite on overlap).
    for c in range(width):
        x_val = x_min + c / (width - 1) * x_span
        y_val = w * x_val + b
        r = to_row(y_val)
        if 0 <= r < height:
            grid[r][c] = "*"

    # Paint the data points.
    for x_val, y_val in zip(xs, ys):
        c = to_col(x_val)
        r = to_row(y_val)
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = "o"

    # Print the grid with axis labels.
    print(f"  y_max = {y_max:.2f}")
    for row in grid:
        print("|" + "".join(row))
    print("+" + "-" * width)
    print(f"  y_min = {y_min:.2f}")
    print(f"  x: {x_min:.2f}" + " " * (width - 14) + f"x: {x_max:.2f}")
    print(f"  learned line: {model.equation()}")
