import unittest
import numpy as np

from src.gsindex import geometrical_separability_index


class TestGeometricalSeparabilityIndex(unittest.TestCase):

    def test_perfect_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample2'])
        expected_value = 1
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(expected_value, actual_value)

    def test_mixed_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample2', 'sample1', 'sample1', 'sample1', 'sample2', 'sample2', 'sample2', 'sample1'])
        expected_value = 0.625
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(expected_value, actual_value)

    def test_no_separation(self):
        input_matrix = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [10, 11], [12, 13], [14, 15], [16, 17]])
        input_labels = np.array(
            ['sample1', 'sample2', 'sample1', 'sample2', 'sample1', 'sample2', 'sample1', 'sample2'])
        expected_value = 0.0
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(expected_value, actual_value)

    def test_single_sample_current_behavior_raises_indexerror(self):
        # Current implementation indexes elements[1] and raises IndexError for single-sample
        input_matrix = np.array([[0.0, 1.0]])
        input_labels = np.array(['a'])
        with self.assertRaises(IndexError):
            geometrical_separability_index(input_matrix, input_labels)

    def test_duplicates_within_class_yield_gsi_one(self):
        # Two identical points per class; nearest neighbor always same class
        input_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        input_labels = np.array(['A', 'A', 'B', 'B'])
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(1.0, actual_value)

    def test_identical_points_across_classes_yield_zero(self):
        # Two identical points with different labels; each nearest neighbor is the other (different class)
        input_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        input_labels = np.array(['A', 'B'])
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(0.0, actual_value)

    def test_multiclass_well_separated_clusters_yield_one(self):
        # Three clusters (A, B, C) far apart; NN stays within class
        input_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
            [20.0, 20.0],
            [20.0, 20.1],
        ])
        input_labels = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(1.0, actual_value)

    def test_numeric_labels_supported(self):
        input_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.0, 1.1],
        ])
        input_labels = np.array([0, 0, 1, 1])
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(1.0, actual_value)

    def test_pandas_inputs_if_available(self):
        try:
            import pandas as pd  # type: ignore
        except Exception:
            self.skipTest("pandas not available")

        input_matrix = pd.DataFrame(
            {
                'x': [0.0, 0.0, 1.0, 1.0],
                'y': [0.0, 0.1, 1.0, 1.1],
            }
        )
        input_labels = pd.Series(['A', 'A', 'B', 'B'])
        actual_value = geometrical_separability_index(input_matrix, input_labels)
        self.assertEqual(1.0, actual_value)

    def test_non_finite_values_dont_crash_and_yield_ratio(self):
        # Presence of NaN/Inf: current implementation does not validate; ensure it returns a ratio
        input_matrix = np.array([
            [0.0, 0.0],
            [np.nan, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
        ])
        input_labels = np.array(['A', 'A', 'B', 'B'])
        value = geometrical_separability_index(input_matrix, input_labels)
        self.assertIsInstance(value, (float, np.floating))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_mismatched_label_length_raises_indexerror(self):
        # Labels shorter than number of samples should index out of bounds
        input_matrix = np.array([
            [0.0, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
        ])
        input_labels = np.array(['A', 'A'])  # length 2, but 3 samples
        with self.assertRaises(IndexError):
            geometrical_separability_index(input_matrix, input_labels)

    def test_non_2d_matrix_raises_valueerror_from_cdist(self):
        # cdist expects 2D arrays; passing 1D should raise ValueError
        input_matrix = np.array([0.0, 1.0, 2.0, 3.0])  # 1D
        input_labels = np.array(['A', 'A', 'B', 'B'])
        with self.assertRaises(ValueError):
            geometrical_separability_index(input_matrix, input_labels)


if __name__ == '__main__':
    unittest.main()
