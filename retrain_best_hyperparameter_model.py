import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load preprocessed data
df = pd.read_csv('train_preprocessed.csv')
X = df.drop(['Personality'], axis=1)
y = df['Personality']

# Encode target
y_map = {k: v for v, k in enumerate(y.unique())}
y_inv_map = {v: k for k, v in y_map.items()}
y_enc = y.map(y_map)

# Best parameters from hyperparameter tuning for LogisticRegression
best_params = {
    'penalty': 'l2',
    'solver': 'saga',
    'C': 0.006548647323524594,
    'max_iter': 424,
    'class_weight': 'balanced',
    'random_state': 42
}

# Train the best model
print("Training best hyperparameter tuned LogisticRegression model...")
best_model = LogisticRegression(**best_params)
best_model.fit(X, y_enc)

# Save the model
joblib.dump(best_model, 'best_model_hyperparameter_tuned.pkl')
print("Best hyperparameter tuned model saved to 'best_model_hyperparameter_tuned.pkl'")

# Test prediction to ensure model works
test_pred = best_model.predict(X[:5])
print(f"Test predictions: {test_pred}")

# Save the inverse mapping for later use
joblib.dump(y_inv_map, 'y_inv_map.pkl')
print("Target inverse mapping saved to 'y_inv_map.pkl'") 