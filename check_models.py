import joblib
import pandas as pd
import numpy as np

# Load both models
model1 = joblib.load('best_model.pkl')
model2 = joblib.load('best_model_hyperparameter_tuned.pkl')

print("Model 1 (best_model.pkl):")
print(f"Type: {type(model1).__name__}")
print(f"Parameters: {model1.get_params()}")
print()

print("Model 2 (best_model_hyperparameter_tuned.pkl):")
print(f"Type: {type(model2).__name__}")
print(f"Parameters: {model2.get_params()}")
print()

# Load test data
X_test = pd.read_csv('test_preprocessed.csv')

# Get predictions from both models
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

# Compare predictions
print(f"Total predictions: {len(pred1)}")
print(f"Predictions that match: {np.sum(pred1 == pred2)}")
print(f"Predictions that differ: {np.sum(pred1 != pred2)}")
print(f"Percentage match: {np.mean(pred1 == pred2) * 100:.2f}%")

# Show some examples where they differ
if np.sum(pred1 != pred2) > 0:
    diff_indices = np.where(pred1 != pred2)[0]
    print(f"\nFirst 5 differences:")
    for i in diff_indices[:5]:
        print(f"Index {i}: Model1={pred1[i]}, Model2={pred2[i]}") 