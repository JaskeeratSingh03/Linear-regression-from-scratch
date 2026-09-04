"""Unit tests for the math in linear_regression.model."""

import unittest

from linear_regression.model import LinearRegression


class TestCost(unittest.TestCase):
    def test_cost_is_zero_when_perfect(self):
        m = LinearRegression()
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(m._compute_cost(y_true, y_pred), 0.0)

    def test_cost_is_positive_when_imperfect(self):
        m = LinearRegression()
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 3.0]  # off by 1 on the first one
        self.assertGreater(m._compute_cost(y_true, y_pred), 0.0)

    def test_cost_matches_formula(self):
        # Hand-checked: predictions [3,3], targets [1,5] -> errors [2,-2] -> sq [4,4] -> sum 8 / (2*2) = 2.0
        m = LinearRegression()
        self.assertAlmostEqual(m._compute_cost([1.0, 5.0], [3.0, 3.0]), 2.0)


class TestPredict(unittest.TestCase):
    def test_predict_returns_one_value_per_row(self):
        # After fit, predict should return one value per row of input.
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [1.0, 3.0, 5.0, 7.0]  # y = 2x + 1
        m = LinearRegression(learning_rate=0.1, n_iters=2000)
        m.fit(X, y)
        preds = m.predict(X)
        self.assertEqual(len(preds), 4)
        for pred, target in zip(preds, y):
            self.assertAlmostEqual(pred, target, delta=0.01)


class TestConvergence(unittest.TestCase):
    def test_recovers_perfect_line_with_no_noise(self):
        # y = 2x + 5 exactly. With enough iterations and a sane learning rate,
        # gradient descent should produce predictions matching the line.
        X = [[float(x)] for x in range(0, 50, 5)]
        y = [2.0 * x[0] + 5.0 for x in X]
        m = LinearRegression(learning_rate=0.1, n_iters=2000)
        m.fit(X, y)
        preds = m.predict(X)
        for pred, target in zip(preds, y):
            self.assertAlmostEqual(pred, target, delta=0.05)

    def test_recovers_noisy_line_close_to_truth(self):
        # With Gaussian noise, we can only get "close enough" — not exact.
        import random
        rng = random.Random(0)
        X = [[float(x)] for x in range(0, 100, 2)]
        y = [2.0 * x[0] + 5.0 + 5.0 * rng.gauss(0, 1) for x in X]
        m = LinearRegression(learning_rate=0.1, n_iters=2000)
        m.fit(X, y)
        preds = m.predict(X)
        # Average prediction should track the true line within a few units.
        for x_row, pred, target in zip(X, preds, y):
            expected = 2.0 * x_row[0] + 5.0
            self.assertAlmostEqual(pred, expected, delta=8.0)

    def test_cost_history_monotonically_decreases(self):
        # Cost should always go down — if it doesn't, gradient descent is broken.
        X = [[float(x)] for x in range(0, 20)]
        y = [3.0 * x[0] + 1.0 for x in X]
        m = LinearRegression(learning_rate=0.05, n_iters=500)
        m.fit(X, y)
        self.assertLess(m.cost_history[-1], m.cost_history[0])
        # And every consecutive step should not increase (allow tiny float wobble).
        for prev, curr in zip(m.cost_history, m.cost_history[1:]):
            self.assertLessEqual(curr, prev + 1e-9)


class TestUntrainedPredict(unittest.TestCase):
    def test_predict_before_fit_returns_bias_only(self):
        # Predicting before fit() is harmless: weights default to [] and bias to 0.0,
        # so every prediction is just the bias term. This documents the behavior
        # so a user knows what to expect if they call predict() too early.
        m = LinearRegression()
        preds = m.predict([[1.0], [2.0]])
        self.assertEqual(preds, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
