# 1. SETUP & IMPORTS
!pip install -q category_encoders xgboost lightgbm catboost optuna shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
import joblib
import shap

# 2. DATA LOADING & INITIAL EXPLORATION

print("=" * 60)
print("PERSONALITY CLASSIFICATION - ENHANCED PIPELINE")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Separate target
y = train['Personality']
X = train.drop(['Personality'], axis=1)
X_test_orig = test.copy()

# Encode target variable
y_map = {k: v for v, k in enumerate(y.unique())}
y_inv_map = {v: k for k, v in y_map.items()}
y_enc = y.map(y_map)

print(f"\nTarget encoding:")
for k, v in y_map.items():
    print(f"  {k} -> {v}")

# 3. EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 3.1 Basic dataset information
print("\nBASIC DATASET INFORMATION")
print("-" * 50)
print("Training data info:")
print(train.info())
print(f"\nMissing values in training data:")
print(train.isnull().sum())
print(f"\nMissing values in test data:")
print(test.isnull().sum())

# 3.2 Target distribution
print(f"\nTARGET DISTRIBUTION")
print("-" * 30)
print(y.value_counts())
print(f"\nClass balance:")
print(y.value_counts(normalize=True))

# 3.3 Histograms of numeric features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# Create EDA output directory
import os
if not os.path.exists('eda_output'):
    os.makedirs('eda_output')

# Histograms
n_features = len(num_cols)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(num_cols):
    axes[idx].hist(X[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[idx].set_title(f'Histogram of {feature}')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')

# Hide unused subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('eda_output/hist_numeric_features.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.4 Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X[num_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Correlation Analysis of Numeric Features')
plt.tight_layout()
plt.savefig('eda_output/corr_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.5 Box plots for outliers
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(num_cols):
    sns.boxplot(y=X[feature], ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Box Plot of {feature}')
    axes[idx].set_xlabel('')

# Hide unused subplots
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('eda_output/boxplots_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.6 Distribution of personality categories (target)
plt.figure(figsize=(8, 6))
sns.countplot(x=y, palette='viridis')
plt.title('Distribution of Personality Categories')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_output/personality_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.7 Pairplot for key numeric features (top 5 by correlation with target)
if len(num_cols) >= 5:
    # Calculate correlation with target
    target_corr = []
    for col in num_cols:
        corr = abs(X[col].corr(y_enc))
        target_corr.append((col, corr))
    
    target_corr.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in target_corr[:5]]
    
    # Create pairplot
    pairplot_data = X[top_features].copy()
    pairplot_data['Personality'] = y
    
    sns.pairplot(pairplot_data, hue='Personality', diag_kind='hist')
    plt.savefig('eda_output/pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save EDA summary
with open('eda_output/summary.txt', 'w') as f:
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset shapes:\n")
    f.write(f"  Train: {train.shape}\n")
    f.write(f"  Test: {test.shape}\n\n")
    f.write(f"Features:\n")
    f.write(f"  Numeric: {num_cols}\n")
    f.write(f"  Categorical: {cat_cols}\n\n")
    f.write(f"Target distribution:\n")
    f.write(str(y.value_counts()) + "\n\n")
    f.write(f"Missing values:\n")
    f.write(str(train.isnull().sum()) + "\n")

# 4. PREPROCESSING

print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# 4.1 Identify columns
drop_cols = ['id']
if 'id' in X.columns:
    X = X.drop('id', axis=1)
if 'id' in X_test_orig.columns:
    X_test_orig = X_test_orig.drop('id', axis=1)

# 4.2 Handle missing values
print("Handling missing values...")
print(f"Missing values in training data: {X.isnull().sum().sum()}")
print(f"Missing values in test data: {X_test_orig.isnull().sum().sum()}")

# 4.3 Numeric imputation
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test = X_test_orig.copy()
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# 4.4 Categorical imputation (only if categorical columns exist)
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# 4.5 Encoding (only if categorical columns exist)
encoders = {}
if len(cat_cols) > 0:
    print("Encoding categorical variables...")
    for col in cat_cols:
        if X[col].nunique() <= 5:
            # One-hot encoding for low cardinality
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(X[[col]])
            ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            ohe_train = pd.DataFrame(ohe.transform(X[[col]]), columns=ohe_cols, index=X.index)
            ohe_test = pd.DataFrame(ohe.transform(X_test[[col]]), columns=ohe_cols, index=X_test.index)

            X = pd.concat([X.drop(col, axis=1), ohe_train], axis=1)
            X_test = pd.concat([X_test.drop(col, axis=1), ohe_test], axis=1)
            encoders[col] = ohe
            print(f"  {col}: One-hot encoded -> {len(ohe_cols)} features")
        else:
            # Target encoding for high cardinality
            te = TargetEncoder()
            te.fit(X[col], y)
            X[col] = te.transform(X[col])
            X_test[col] = te.transform(X_test[col])
            encoders[col] = te
            print(f"  {col}: Target encoded")

# 4.6 Scaling
print("Scaling numeric features...")
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 4.7 Save preprocessed data
X['Personality'] = y
X.to_csv('train_preprocessed.csv', index=False)
X_test.to_csv('test_preprocessed.csv', index=False)

# Save preprocessing objects
joblib.dump(num_imputer, 'num_imputer.pkl')
if len(cat_cols) > 0:
    joblib.dump(cat_imputer, 'cat_imputer.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(y_inv_map, 'y_inv_map.pkl')

print("Preprocessing completed and saved!")

# 5. MODEL TRAINING & EVALUATION

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

# 5.1 Split data for evaluation
X_train, X_val, y_train, y_val = train_test_split(X.drop('Personality', axis=1), y_enc, 
                                                   test_size=0.2, random_state=42, stratify=y_enc)

# 5.2 Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=42),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                             n_jobs=-1, random_state=42),
    'LightGBM': LGBMClassifier(n_jobs=-1, random_state=42, verbose=0),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

# 5.3 Hyperparameter grids
param_grids = {
    'LogisticRegression': {'C': [0.1, 1, 10], 'class_weight': [None, 'balanced']},
    'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
    'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.3]},
    'LightGBM': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.3]},
    'CatBoost': {'iterations': [50, 100], 'depth': [3, 5], 'learning_rate': [0.1, 0.3]}
}

# 5.4 Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5.5 Train and evaluate models
raw_results = {}
best_estimators = {}
train_accuracies = {}
test_accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # GridSearch for hyperparameter tuning
    gs = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    
    # Train and test predictions
    y_train_pred = best.predict(X_train)
    y_val_pred = best.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_val, y_val_pred)
    
    # Store results
    raw_results[name] = {
        'best_params': gs.best_params_,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1_score(y_val, y_val_pred, average='weighted'),
        'precision': precision_score(y_val, y_val_pred, average='weighted'),
        'recall': recall_score(y_val, y_val_pred, average='weighted')
    }
    
    best_estimators[name] = best
    train_accuracies[name] = train_acc
    test_accuracies[name] = test_acc
    
    print(f"{name:15s} | CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# 5.6 Create results DataFrame
results_df = pd.DataFrame(raw_results).T
results_df = results_df.sort_values('test_accuracy', ascending=False)

print(f"\nFINAL MODEL RANKINGS")
print("=" * 60)
print(results_df[['cv_accuracy', 'train_accuracy', 'test_accuracy', 'f1_score']].round(4))

# Save results
results_df.to_csv('model_comparison.csv')

# 6. VISUALIZATION - ACCURACY COMPARISON

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

# 6.1 Create accuracy comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training vs Test Accuracy
models_list = list(train_accuracies.keys())
train_acc_list = [train_accuracies[model] for model in models_list]
test_acc_list = [test_accuracies[model] for model in models_list]

x = np.arange(len(models_list))
width = 0.35

ax1.bar(x - width/2, train_acc_list, width, label='Training Accuracy', alpha=0.8, color='skyblue')
ax1.bar(x + width/2, test_acc_list, width, label='Test Accuracy', alpha=0.8, color='lightcoral')

ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy')
ax1.set_title('Training vs Test Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models_list, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, (train_acc, test_acc) in enumerate(zip(train_acc_list, test_acc_list)):
    ax1.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', ha='center', va='bottom')
    ax1.text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom')

# 6.2 Detailed metrics comparison
metrics_df = results_df[['cv_accuracy', 'train_accuracy', 'test_accuracy', 'f1_score', 'precision', 'recall']]
metrics_df.plot(kind='bar', ax=ax2, figsize=(10, 6))
ax2.set_title('Detailed Model Performance Metrics')
ax2.set_xlabel('Models')
ax2.set_ylabel('Score')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.3 Confusion Matrix for Best Model
best_model_name = results_df['test_accuracy'].idxmax()
best_model = best_estimators[best_model_name]

y_val_pred_best = best_model.predict(X_val)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(y_map.keys()), 
            yticklabels=list(y_map.keys()), ax=ax1)
ax1.set_title(f'Confusion Matrix - {best_model_name}')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Classification Report
report = classification_report(y_val, y_val_pred_best, 
                             target_names=list(y_map.keys()), 
                             output_dict=True)
report_df = pd.DataFrame(report).transpose()

sns.heatmap(report_df.iloc[:-3, :].astype(float), annot=True, fmt='.3f', 
            cmap='YlOrRd', ax=ax2)
ax2.set_title(f'Classification Report - {best_model_name}')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. FEATURE IMPORTANCE

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    print(f"Analyzing feature importance for {best_model_name}...")
    
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    
    # Create feature importance DataFrame
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = fi_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Top 10 most important features:")
    print(fi_df.head(10)[['feature', 'importance']].to_string(index=False))

else:
    print(f"Feature importance not available for {best_model_name}")
    # Try SHAP for model interpretability
    try:
        print("Attempting SHAP analysis...")
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig('feature_importance_shap.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

# 8. PREDICTION & SUBMISSION

print("\n" + "=" * 60)
print("PREDICTION & SUBMISSION")
print("=" * 60)

print(f"Making predictions with best model: {best_model_name}")

# 8.1 Ensure X_test has the same features as training data
print("Aligning test features with training features...")

# Get the feature names used during training
train_features = X_train.columns.tolist()
test_features = X_test.columns.tolist()

print(f"Training features: {len(train_features)}")
print(f"Test features: {len(test_features)}")

# Find missing features in test set
missing_features = set(train_features) - set(test_features)
extra_features = set(test_features) - set(train_features)

if missing_features:
    print(f"Missing features in test set: {missing_features}")
    # Add missing features with zeros
    for feature in missing_features:
        X_test[feature] = 0

if extra_features:
    print(f"Extra features in test set: {extra_features}")
    # Remove extra features
    X_test = X_test.drop(columns=list(extra_features))

# Ensure same column order as training
X_test = X_test[train_features]

print(f"Test set now has {X_test.shape[1]} features (same as training)")

# 8.2 Make predictions on test set
y_test_pred = best_model.predict(X_test)

# 8.3 Decode predictions
y_test_pred_label = [y_inv_map.get(i, i) for i in y_test_pred]

# 8.4 Create submission
sub = pd.DataFrame({'id': test['id'], 'Personality': y_test_pred_label})
sub.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")

# 8.5 Display prediction statistics
print(f"\nPREDICTION STATISTICS")
print("-" * 40)
print(f"Total predictions: {len(y_test_pred_label)}")
print(f"Prediction distribution:")
pred_dist = pd.Series(y_test_pred_label).value_counts()
for personality, count in pred_dist.items():
    print(f"  {personality}: {count} ({count/len(y_test_pred_label)*100:.1f}%)")

# 8.6 Show first few predictions
print(f"\nFIRST 10 PREDICTIONS")
print("-" * 40)
print(sub.head(10))

# 8.7 Verify feature alignment
print(f"\nFEATURE ALIGNMENT VERIFICATION")
print("-" * 40)
print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Features match: {list(X_train.columns) == list(X_test.columns)}")

# 9. SAVE BEST MODEL

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model ({best_model_name}) saved as 'best_model.pkl'")

# Save all preprocessing objects
print("Preprocessing objects saved:")
print("- num_imputer.pkl")
print("- cat_imputer.pkl") 
print("- encoders.pkl")
print("- scaler.pkl")
print("- y_inv_map.pkl")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("Files created:")
print("- train_preprocessed.csv")
print("- test_preprocessed.csv")
print("- model_comparison.csv")
print("- best_model.pkl")
print("- submission.csv")
print("- Various visualization files (.png)")
print("- eda_output/ directory with EDA results") 