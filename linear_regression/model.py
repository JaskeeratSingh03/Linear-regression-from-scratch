"""Linear regression via gradient descent — pure Python, no numpy.

The math, in three equations:

    y_pred[i] = w * x[i] + b

    MSE = (1 / 2n) * sum (y_pred[i] - y[i]) ** 2

    w := w - alpha * dw        where  dw = (1/n) * sum (y_pred[i] - y[i]) * x[i]
    b := b - alpha * db        where  db = (1/n) * sum (y_pred[i] - y[i])

Data shapes:
    X: list of lists — X[i][j] is feature j of sample i (so we can grow to N features)
    y: list of floats — one target per sample
"""


class LinearRegression:
    """A line y = w * x + b learned by gradient descent."""

    def __init__(self, learning_rate=0.01, n_iters=1000):
        # alpha — how big a step we take on each update.
        self.lr = learning_rate
        # how many times we run the predict -> cost -> update loop.
        self.n_iters = n_iters
        # learned parameters — populated during fit().
        # Defaulting to [] (not None) so predict() works before fit() and returns
        # just the bias term.
        self.weights = []
        self.bias = 0.0
        # MSE after every iteration — useful for the cost-curve plot.
        self.cost_history = []

    def fit(self, X, y):
        """Learn weights and bias by running gradient descent `n_iters` times."""
        n_samples = len(X)
        n_features = len(X[0])

        # Feature scaling: standardize each feature to mean 0, std 1.
        # This keeps gradients well-behaved so a learning_rate of 0.01–0.1 works
        # regardless of the original input scale (e.g. x in [0, 100] vs [0, 1]).
        # The scaling factors are saved so predict() can apply the same transform.
        self._feature_means = [sum(row[j] for row in X) / n_samples
                                for j in range(n_features)]
        self._feature_stds = []
        for j in range(n_features):
            mean = self._feature_means[j]
            variance = sum((row[j] - mean) ** 2 for row in X) / n_samples
            self._feature_stds.append(variance ** 0.5 or 1.0)

        X_scaled = [
            [(row[j] - self._feature_means[j]) / self._feature_stds[j]
             for j in range(n_features)]
            for row in X
        ]

        # Also scale y so the bias update is well-behaved.
        # Predictions come back in scaled-y space and we unscale at predict() time.
        self._y_mean = sum(y) / n_samples
        y_var = sum((yi - self._y_mean) ** 2 for yi in y) / n_samples
        self._y_std = y_var ** 0.5 or 1.0
        y_scaled = [(yi - self._y_mean) / self._y_std for yi in y]

        # Start at zero. We'll improve from here.
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iters):
            # 1) Make a prediction with current weights (on scaled inputs).
            y_pred_scaled = self._predict_scaled(X_scaled)

            # 2) Measure how wrong we are (in scaled space).
            cost = self._compute_cost(y_scaled, y_pred_scaled)
            self.cost_history.append(cost)

            # 3) Figure out which direction to nudge w and b.
            dw, db = self._compute_gradients(X_scaled, y_scaled, y_pred_scaled)

            # 4) Take a small step in that direction.
            self.weights = [w - self.lr * dw[j] for j, w in enumerate(self.weights)]
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        """Return predictions on the original (unscaled) feature/target space."""
        if not self.weights:
            # Model hasn't been fit yet — return the bias for every row.
            return [self.bias for _ in X]
        X_scaled = [
            [(row[j] - self._feature_means[j]) / self._feature_stds[j]
             for j in range(len(self.weights))]
            for row in X
        ]
        # Predict in scaled space, then unscale back to original y.
        y_pred_scaled = self._predict_scaled(X_scaled)
        return [yp * self._y_std + self._y_mean for yp in y_pred_scaled]

    def _predict_scaled(self, X_scaled):
        """Internal: predict using already-scaled features (output is scaled y)."""
        return [sum(w * x_j for w, x_j in zip(self.weights, row)) + self.bias
                for row in X_scaled]

    def equation(self):
        """Return the learned equation in the ORIGINAL (unscaled) feature space.

        Useful for the demo menu — the user sees w and b in the units their data is in,
        not in the units we used internally for numerical stability.
        Returns a string like "y = 2.44 * x + 11.91" (single-feature models only).
        """
        if not self.weights or len(self.weights) != 1:
            return "(equation display only supports single-feature models)"
        w_orig = self.weights[0] * self._y_std / self._feature_stds[0]
        b_orig = self.bias * self._y_std + self._y_mean - w_orig * self._feature_means[0]
        return f"y = {w_orig:.4f} * x + {b_orig:.4f}"

    def coefficients_original_scale(self):
        """Return (w_orig, b_orig) in the original feature space (single-feature only)."""
        if not self.weights or len(self.weights) != 1:
            return None
        w_orig = self.weights[0] * self._y_std / self._feature_stds[0]
        b_orig = self.bias * self._y_std + self._y_mean - w_orig * self._feature_means[0]
        return w_orig, b_orig

    def _compute_cost(self, y_true, y_pred):
        """Mean Squared Error, with the (1/2) trick to cancel on differentiation."""
        n = len(y_true)
        total = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true))
        return total / (2 * n)

    def _compute_gradients(self, X, y_true, y_pred):
        """Partial derivatives of MSE w.r.t. each weight and the bias."""
        n = len(y_true)
        n_features = len(X[0])
        dw = [0.0] * n_features
        db = 0.0
        for i in range(n):
            error = y_pred[i] - y_true[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error
        dw = [d / n for d in dw]
        db = db / n
        return dw, db
