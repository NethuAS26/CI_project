import pandas as pd
import joblib
import numpy as np

# Load test data
X_test = pd.read_csv('test_preprocessed.csv')

# Load hyperparameter tuned model and encoders
model = joblib.load('best_model_hyperparameter_tuned.pkl')
encoders = joblib.load('encoders.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Load original test for id
orig_test = pd.read_csv('test.csv')
ids = orig_test['id']

# Predict
if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_test)
    np.save('test_pred_proba_hyperparameter_tuned.npy', y_pred_proba)
y_pred = model.predict(X_test)

# If target was encoded, decode
try:
    y_inv_map = joblib.load('y_inv_map.pkl')
    y_pred_label = [y_inv_map.get(i, i) for i in y_pred]
except:
    y_pred_label = y_pred

# Prepare submission
sub = pd.DataFrame({'id': ids, 'Personality': y_pred_label})
sub.to_csv('submission_hyperparameter_tuned.csv', index=False)
print('Hyperparameter tuned predictions saved to submission_hyperparameter_tuned.csv')

# Display first few predictions
print("\nFirst 10 predictions:")
print(sub.head(10))

# Display prediction distribution
print(f"\nPrediction distribution:")
print(sub['Personality'].value_counts()) 