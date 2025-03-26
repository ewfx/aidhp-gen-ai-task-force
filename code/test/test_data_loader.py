import unittest
import pandas as pd
from app import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)  # Ensure data is not empty

if __name__ == '__main__':
    unittest.main()
