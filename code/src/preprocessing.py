import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    """
    Preprocesses customer financial data.

    Args:
        data (pd.DataFrame): Raw customer financial data.

    Returns:
        tuple: Processed feature and target sets (X_train, X_test, y_train, y_test).
    """
    features = data.drop(columns=["customer_id", "recommended_product"])
    target = data["recommended_product"]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
