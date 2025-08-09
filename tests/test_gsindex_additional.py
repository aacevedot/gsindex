import unittest
import numpy as np

from src.gsindex import geometrical_separability_index


class TestGeometricalSeparabilityIndexAdditional(unittest.TestCase):

    def test_multiclass_perfect_separation(self):
        # Three well-separated clusters with distinct labels
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [-0.1, 0.0],   # class A
            [5.0, 5.0], [5.1, 5.0], [4.9, 5.0],     # class B
            [-5.0, 5.0], [-5.1, 5.0], [-4.9, 5.0],  # class C
        ])
        y = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        gsi = geometrical_separability_index(X, y)
        self.assertEqual(1.0, gsi)

    def test_numeric_labels(self):
        # Two clusters with integer labels
        X = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        gsi = geometrical_separability_index(X, y)
        self.assertEqual(1.0, gsi)

    def test_output_range_and_type(self):
        # Mixed separation should yield value in [0, 1] and be float-like
        X = np.array([
            [0, 0], [1, 0], [0, 1],
            [3, 3], [4, 3], [3, 4],
        ])
        y = np.array(['A', 'B', 'A', 'B', 'B', 'A'])
        gsi = geometrical_separability_index(X, y)
        self.assertTrue(0.0 <= gsi <= 1.0)
        self.assertIsInstance(gsi, (float, np.floating))

    def test_single_sample_raises(self):
        # With a single sample, nearest neighbor beyond self is undefined
        X = np.array([[0.0, 0.0]])
        y = np.array(['A'])
        with self.assertRaises(IndexError):
            _ = geometrical_separability_index(X, y)

    def test_mismatched_label_length_raises(self):
        # Labels length must match number of samples
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array(['A', 'A'])  # shorter than samples
        with self.assertRaises(Exception):
            _ = geometrical_separability_index(X, y)

