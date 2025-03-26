from app import load_data, preprocess_data, train_model, generate_recommendations

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    recommendations = generate_recommendations(model, X_test)

    print("Generated Product Recommendations:", recommendations)
