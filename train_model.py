import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
df = pd.read_csv('train_preprocessed.csv')
X = df.drop(['Personality'], axis=1)
y = df['Personality']

# Encode target
y_map = {k: v for v, k in enumerate(y.unique())}
y_inv_map = {v: k for k, v in y_map.items()}
y_enc = y.map(y_map)

# Model configs
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(verbose=0)
}

param_grids = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'DecisionTree': {'max_depth': [3, 5, 10]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
    'LightGBM': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
    'CatBoost': {'iterations': [50, 100], 'depth': [3, 5]}
}

results = {}
best_score = 0
best_model = None
best_name = None
best_y_pred = None

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f'\nTraining {name}...')
    if name in ['XGBoost', 'LightGBM', 'CatBoost']:
        # Use Optuna for boosting models
        def objective(trial):
            params = {}
            for k, v in param_grids[name].items():
                params[k] = trial.suggest_categorical(k, v)
            m = model.__class__(**params)
            scores = []
            for train_idx, val_idx in skf.split(X, y_enc):
                m.fit(X.iloc[train_idx], y_enc.iloc[train_idx])
                y_pred = m.predict(X.iloc[val_idx])
                scores.append(accuracy_score(y_enc.iloc[val_idx], y_pred))
            return np.mean(scores)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        best_params = study.best_params
        model.set_params(**best_params)
    else:
        gs = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
        gs.fit(X, y_enc)
        model = gs.best_estimator_
    # Cross-validated metrics
    scores = []
    f1s = []
    aucs = []
    y_true_all = []
    y_pred_all = []
    for train_idx, val_idx in skf.split(X, y_enc):
        model.fit(X.iloc[train_idx], y_enc.iloc[train_idx])
        y_pred = model.predict(X.iloc[val_idx])
        y_true_all.extend(y_enc.iloc[val_idx])
        y_pred_all.extend(y_pred)
        scores.append(accuracy_score(y_enc.iloc[val_idx], y_pred))
        f1s.append(f1_score(y_enc.iloc[val_idx], y_pred, average='weighted'))
        try:
            aucs.append(roc_auc_score(y_enc.iloc[val_idx], model.predict_proba(X.iloc[val_idx]), multi_class='ovr'))
        except:
            aucs.append(np.nan)
    acc = np.mean(scores)
    f1 = np.mean(f1s)
    auc = np.nanmean(aucs)
    results[name] = {'accuracy': acc, 'f1': f1, 'roc_auc': auc}
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = name
        best_y_pred = y_pred_all

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('model_comparison.csv')

# Plot model comparison
plt.figure(figsize=(8,5))
results_df[['accuracy', 'f1', 'roc_auc']].plot(kind='bar')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Confusion matrix for best model
cm = confusion_matrix(y_enc, best_y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y_map.keys(), yticklabels=y_map.keys())
plt.title(f'Confusion Matrix ({best_name})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance (tree-based or SHAP)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_names = X.columns
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    fi[:20].plot(kind='barh')
    plt.title(f'Feature Importance ({best_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
else:
    explainer = shap.Explainer(best_model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig('feature_importance_shap.png')
    plt.close()

# Save best model
joblib.dump(best_model, 'best_model.pkl')
print(f'Best model: {best_name} (accuracy={best_score:.4f}) saved as best_model.pkl') 