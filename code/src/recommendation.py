import numpy as np

def generate_recommendations(model, X_test):
    """
    Generates product recommendations based on the trained model.

    Args:
        model (keras.Model): Trained AI model.
        X_test (np.array): Test features.

    Returns:
        np.array: Recommended products.
    """
    predictions = model.predict(X_test)
    recommended_products = np.argmax(predictions, axis=1)
    return recommended_products
