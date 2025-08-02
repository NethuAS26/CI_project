import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Advanced objective functions
def objective_ensemble_voting(trial):
    # Define base models with hyperparameters
    lr_params = {
        'C': trial.suggest_float('lr_C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
        'solver': 'liblinear'
    }
    
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
    }
    
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
    }
    
    # Create base models
    lr = LogisticRegression(**lr_params, random_state=42)
    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    
    # Voting weights
    weights = [
        trial.suggest_float('weight_lr', 0.1, 1.0),
        trial.suggest_float('weight_rf', 0.1, 1.0),
        trial.suggest_float('weight_xgb', 0.1, 1.0)
    ]
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)],
        voting='soft',
        weights=weights
    )
    
    scores = cross_val_score(voting_clf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_ensemble_stacking(trial):
    # Base models
    lr_params = {
        'C': trial.suggest_float('lr_C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('lr_penalty', ['l1', 'l2']),
        'solver': 'liblinear'
    }
    
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 15)
    }
    
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
    }
    
    # Meta-learner
    meta_params = {
        'C': trial.suggest_float('meta_C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('meta_penalty', ['l1', 'l2']),
        'solver': 'liblinear'
    }
    
    # Create models
    lr = LogisticRegression(**lr_params, random_state=42)
    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    meta_learner = LogisticRegression(**meta_params, random_state=42)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)],
        final_estimator=meta_learner,
        cv=3
    )
    
    scores = cross_val_score(stacking_clf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_feature_selection(trial):
    # Feature selection method
    selection_method = trial.suggest_categorical('selection_method', ['kbest', 'rfe'])
    
    if selection_method == 'kbest':
        k = trial.suggest_int('k_features', 5, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
    else:  # RFE
        n_features = trial.suggest_int('n_features', 5, X.shape[1])
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
    
    # Model parameters
    model_type = trial.suggest_categorical('model_type', ['lr', 'rf', 'xgb'])
    
    if model_type == 'lr':
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear'
        }
        model = LogisticRegression(**params, random_state=42)
    elif model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 15)
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    else:  # xgb
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    
    # Apply feature selection
    X_selected = selector.fit_transform(X, y_enc)
    
    scores = cross_val_score(model, X_selected, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_bagging(trial):
    # Base estimator
    base_estimator_type = trial.suggest_categorical('base_estimator', ['dt', 'lr'])
    
    if base_estimator_type == 'dt':
        base_params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        base_estimator = DecisionTreeClassifier(**base_params, random_state=42)
    else:
        base_params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear'
        }
        base_estimator = LogisticRegression(**base_params, random_state=42)
    
    # Bagging parameters
    bagging_params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'bootstrap_features': trial.suggest_categorical('bootstrap_features', [True, False])
    }
    
    bagging_clf = BaggingClassifier(
        base_estimator=base_estimator,
        **bagging_params,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(bagging_clf, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

def objective_multi_objective(trial):
    # This is a simplified multi-objective approach
    # In practice, you might want to use Optuna's multi-objective optimization
    
    # Model parameters
    model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgb'])
    
    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif model_type == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    else:  # lgb
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    
    # Calculate multiple metrics
    accuracy_scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model, X, y_enc, cv=5, scoring='f1_weighted', n_jobs=-1)
    
    # Return weighted combination (you can adjust weights)
    return 0.7 * accuracy_scores.mean() + 0.3 * f1_scores.mean()

# Run advanced hyperparameter optimization
advanced_config = {
    'Ensemble_Voting': objective_ensemble_voting,
    'Ensemble_Stacking': objective_ensemble_stacking,
    'Feature_Selection': objective_feature_selection,
    'Bagging': objective_bagging,
    'Multi_Objective': objective_multi_objective
}

advanced_results = {}
advanced_models = {}

print("Starting advanced hyperparameter tuning...")
print("=" * 60)

for model_name, objective_func in advanced_config.items():
    print(f"\nOptimizing {model_name}...")
    
    # Create study with pruning
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Optimize with more trials for complex methods
    study.optimize(objective_func, n_trials=30, show_progress_bar=True)
    
    # Store results
    advanced_results[model_name] = {
        'best_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }
    
    print(f"Best {model_name} score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

# Save advanced results
advanced_results_df = pd.DataFrame(advanced_results).T
advanced_results_df.to_csv('advanced_tuning_results.csv')

# Create comparison plot
plt.figure(figsize=(14, 8))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Basic models comparison (from previous tuning)
try:
    basic_results = pd.read_csv('hyperparameter_tuning_results.csv', index_col=0)
    basic_models = basic_results.index.tolist()
    basic_scores = basic_results['best_score'].tolist()
    
    bars1 = ax1.bar(basic_models, basic_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Basic Hyperparameter Tuning Results', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Score', fontsize=10)
    ax1.set_ylim(0.96, 0.97)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars1, basic_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=8)
except:
    ax1.text(0.5, 0.5, 'Basic results not available', ha='center', va='center', transform=ax1.transAxes)

# Advanced methods comparison
advanced_models = list(advanced_results.keys())
advanced_scores = [advanced_results[model]['best_score'] for model in advanced_models]
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

bars2 = ax2.bar(advanced_models, advanced_scores, color=colors, alpha=0.7)
ax2.set_title('Advanced Hyperparameter Tuning Results', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy Score', fontsize=10)
ax2.set_ylim(0.96, 0.97)
ax2.tick_params(axis='x', rotation=45)

# Add value labels
for bar, score in zip(bars2, advanced_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
            f'{score:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('advanced_tuning_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Find best advanced model
best_advanced_model = max(advanced_results.items(), key=lambda x: x[1]['best_score'])
print(f"\n" + "=" * 60)
print(f"BEST ADVANCED MODEL: {best_advanced_model[0]}")
print(f"Best Accuracy: {best_advanced_model[1]['best_score']:.4f}")
print(f"Best Parameters: {best_advanced_model[1]['best_params']}")

# Create detailed advanced report
with open('advanced_tuning_report.txt', 'w') as f:
    f.write("ADVANCED HYPERPARAMETER TUNING REPORT\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Advanced Methods Tested:\n")
    f.write("1. Ensemble Voting - Combines multiple models with weighted voting\n")
    f.write("2. Ensemble Stacking - Uses meta-learner to combine base models\n")
    f.write("3. Feature Selection - Optimizes feature subset selection\n")
    f.write("4. Bagging - Bootstrap aggregating with hyperparameter tuning\n")
    f.write("5. Multi-Objective - Balances accuracy and F1-score\n\n")
    
    for model_name, result in advanced_results.items():
        f.write(f"{model_name}:\n")
        f.write(f"  Best Accuracy: {result['best_score']:.4f}\n")
        f.write(f"  Trials: {result['n_trials']}\n")
        f.write(f"  Best Parameters:\n")
        for param, value in result['best_params'].items():
            f.write(f"    {param}: {value}\n")
        f.write("\n")
    
    f.write(f"\nOVERALL BEST ADVANCED: {best_advanced_model[0]} ({best_advanced_model[1]['best_score']:.4f})\n")

print(f"\nAdvanced results saved to:")
print("- advanced_tuning_results.csv")
print("- advanced_tuning_comparison.png") 
print("- advanced_tuning_report.txt") 