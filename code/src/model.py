import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_model(X_train, y_train):
    """
    Trains an AI model for customer financial product recommendation.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.

    Returns:
        keras.Model: Trained model.
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(set(y_train)), activation='softmax')  # Multi-class classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model
