"""Unit tests for linear_regression.dataset."""

import os
import tempfile
import unittest

from linear_regression.dataset import load_csv, make_synthetic


class TestSynthetic(unittest.TestCase):
    def test_returns_correct_lengths(self):
        X, y = make_synthetic(n_samples=30)
        self.assertEqual(len(X), 30)
        self.assertEqual(len(y), 30)

    def test_each_x_row_has_one_feature(self):
        X, _ = make_synthetic(n_samples=10)
        for row in X:
            self.assertEqual(len(row), 1)

    def test_is_reproducible_with_seed(self):
        X1, y1 = make_synthetic(n_samples=20, seed=7)
        X2, y2 = make_synthetic(n_samples=20, seed=7)
        self.assertEqual(X1, X2)
        self.assertEqual(y1, y2)

    def test_different_seeds_yield_different_data(self):
        X1, y1 = make_synthetic(n_samples=20, seed=1)
        X2, y2 = make_synthetic(n_samples=20, seed=2)
        self.assertNotEqual(X1, X2)

    def test_zero_noise_means_perfectly_linear(self):
        X, y = make_synthetic(n_samples=10, noise=0.0, true_w=3.0, true_b=7.0, seed=1)
        for x_row, y_val in zip(X, y):
            self.assertAlmostEqual(3.0 * x_row[0] + 7.0, y_val)


class TestLoadCSV(unittest.TestCase):
    def test_reads_sample_file(self):
        # Resolve relative to the project root, not cwd.
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        path = os.path.join(project_root, "data", "sample.csv")
        X, y = load_csv(path)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        for row in X:
            self.assertEqual(len(row), 1)

    def test_blank_lines_are_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y\n")
            f.write("1.0,3.0\n")
            f.write("\n")  # blank line
            f.write("2.0,5.0\n")
            path = f.name
        try:
            X, y = load_csv(path)
            self.assertEqual(len(X), 2)
            self.assertEqual(y, [3.0, 5.0])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
