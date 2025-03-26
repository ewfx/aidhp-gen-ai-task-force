import unittest
import numpy as np
from app import load_data, preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(data)

        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertEqual(X_train.shape[1], X_test.shape[1])  # Ensure feature dimensions match

if __name__ == '__main__':
    unittest.main()
