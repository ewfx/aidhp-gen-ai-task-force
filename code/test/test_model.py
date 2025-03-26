import unittest
from app import load_data, preprocess_data, train_model

class TestModel(unittest.TestCase):
    def test_train_model(self):
        data = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)

        self.assertTrue(hasattr(model, "predict"))  # Check if model has predict method

if __name__ == '__main__':
    unittest.main()
