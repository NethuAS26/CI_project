import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import joblib
import os

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separate target
y = train['Personality']
X = train.drop(['Personality'], axis=1)
X_test = test.copy()

# Identify columns
drop_cols = ['id']
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Impute numeric
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
joblib.dump(num_imputer, 'num_imputer.pkl')

# Impute categorical
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
joblib.dump(cat_imputer, 'cat_imputer.pkl')

# Encoding
encoded_X = X.copy()
encoded_X_test = X_test.copy()
encoders = {}
for col in cat_cols:
    if X[col].nunique() <= 5:
        # One-hot encoding
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(X[[col]])
        ohe_cols = [f'{col}_{cat}' for cat in ohe.categories_[0]]
        ohe_train = pd.DataFrame(ohe.transform(X[[col]]), columns=ohe_cols, index=X.index)
        ohe_test = pd.DataFrame(ohe.transform(X_test[[col]]), columns=ohe_cols, index=X_test.index)
        encoded_X = pd.concat([encoded_X.drop(col, axis=1), ohe_train], axis=1)
        encoded_X_test = pd.concat([encoded_X_test.drop(col, axis=1), ohe_test], axis=1)
        encoders[col] = ohe
    else:
        # Target encoding
        te = TargetEncoder()
        te.fit(X[col], y)
        encoded_X[col] = te.transform(X[col])
        encoded_X_test[col] = te.transform(X_test[col])
        encoders[col] = te
joblib.dump(encoders, 'encoders.pkl')

# Scale numeric features
scaler = StandardScaler()
encoded_X[num_cols] = scaler.fit_transform(encoded_X[num_cols])
encoded_X_test[num_cols] = scaler.transform(encoded_X_test[num_cols])
joblib.dump(scaler, 'scaler.pkl')

# Save preprocessed data
encoded_X['Personality'] = y
encoded_X.to_csv('train_preprocessed.csv', index=False)
encoded_X_test.to_csv('test_preprocessed.csv', index=False) 