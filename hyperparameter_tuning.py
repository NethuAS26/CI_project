import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
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

# Define objective functions for each model
def objective_logistic(trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    
    # Ensure solver compatibility with penalty
    if penalty == 'elasticnet':
        solver = 'saga'
    elif penalty == 'l1':
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    else:  # l2
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': penalty,
        'solver': solver,
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
    }
    
    # Add l1_ratio only for elasticnet
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    model = LogisticRegression(**params, random_state=42)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_decision_tree(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    
    model = DecisionTreeClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_random_forest(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params['bootstrap'] else None
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 3)  # 1=manhattan, 2=euclidean, 3=minkowski
    }
    
    model = KNeighborsClassifier(**params, n_jobs=-1)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 3)
    }
    
    model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_lightgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.5) if trial.params['boosting_type'] == 'dart' else None
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50)
    }
    # Only set one of class_weights or auto_class_weights
    if trial.suggest_categorical('use_class_weights', [True, False]):
        params['class_weights'] = [1, 2.8]
    else:
        params['auto_class_weights'] = 'Balanced'
    model = cb.CatBoostClassifier(**params, random_state=42, verbose=0)
    scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run hyperparameter optimization for each model
models_config = {
    'LogisticRegression': objective_logistic,
    'DecisionTree': objective_decision_tree,
    'RandomForest': objective_random_forest,
    'KNN': objective_knn,
    'XGBoost': objective_xgboost,
    'LightGBM': objective_lightgbm,
    'CatBoost': objective_catboost
}

results = {}
best_models = {}

print("Starting comprehensive hyperparameter tuning...")
print("=" * 60)

for model_name, objective_func in models_config.items():
    print(f"\nOptimizing {model_name}...")
    
    # Create study with pruning
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize with more trials for better results
    study.optimize(objective_func, n_trials=50, show_progress_bar=True)
    
    # Store results
    results[model_name] = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    # Train best model
    best_params = study.best_params
    if model_name == 'LogisticRegression':
        best_model = LogisticRegression(**best_params, random_state=42)
    elif model_name == 'DecisionTree':
        best_model = DecisionTreeClassifier(**best_params, random_state=42)
    elif model_name == 'RandomForest':
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'KNN':
        best_model = KNeighborsClassifier(**best_params, n_jobs=-1)
    elif model_name == 'XGBoost':
        best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    elif model_name == 'LightGBM':
        best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    elif model_name == 'CatBoost':
        # Remove use_class_weights parameter as it's not a valid CatBoost parameter
        catboost_params = {k: v for k, v in best_params.items() if k != 'use_class_weights'}
        best_model = cb.CatBoostClassifier(**catboost_params, random_state=42, verbose=0)
    
    best_models[model_name] = best_model
    
    print(f"Best {model_name} score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('hyperparameter_tuning_results.csv')

# Create comparison plot
plt.figure(figsize=(12, 6))
models = list(results.keys())
scores = [results[model]['best_score'] for model in models]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

bars = plt.bar(models, scores, color=colors)
plt.title('Hyperparameter Tuning Results - Best Cross-Validation Accuracy', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.ylim(0.96, 0.97)
plt.xticks(rotation=45)

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('hyperparameter_tuning_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Find best overall model
best_overall_model = max(results.items(), key=lambda x: x[1]['best_score'])
print(f"\n" + "=" * 60)
print(f"BEST OVERALL MODEL: {best_overall_model[0]}")
print(f"Best Accuracy: {best_overall_model[1]['best_score']:.4f}")
print(f"Best Parameters: {best_overall_model[1]['best_params']}")

# Save best model
best_model_name = best_overall_model[0]
best_model_instance = best_models[best_model_name]
joblib.dump(best_model_instance, f'best_model_hyperparameter_tuned.pkl')

# Create detailed results report
with open('hyperparameter_tuning_report.txt', 'w') as f:
    f.write("COMPREHENSIVE HYPERPARAMETER TUNING REPORT\n")
    f.write("=" * 50 + "\n\n")
    
    for model_name, result in results.items():
        f.write(f"{model_name}:\n")
        f.write(f"  Best Accuracy: {result['best_score']:.4f}\n")
        f.write(f"  Trials: {result['n_trials']}\n")
        f.write(f"  Best Parameters:\n")
        for param, value in result['best_params'].items():
            f.write(f"    {param}: {value}\n")
        f.write("\n")
    
    f.write(f"\nOVERALL BEST: {best_overall_model[0]} ({best_overall_model[1]['best_score']:.4f})\n")

print(f"\nResults saved to:")
print("- hyperparameter_tuning_results.csv")
print("- hyperparameter_tuning_comparison.png") 
print("- hyperparameter_tuning_report.txt")
print("- best_model_hyperparameter_tuned.pkl") 