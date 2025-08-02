# Implementation Details for Personality Classification
# This script documents the complete implementation pipeline with detailed explanations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def document_data_preprocessing():
    """Document the data preprocessing pipeline"""
    print("=" * 60)
    print("DATA PREPROCESSING IMPLEMENTATION DETAILS")
    print("=" * 60)
    
    print("\n1. DATA LOADING AND INITIAL EXPLORATION")
    print("-" * 50)
    print("""
    The implementation begins with loading the training and test datasets:
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    Key observations:
    - Training data: 9,156 samples with personality labels
    - Test data: 3,076 samples for prediction
    - Features: Mix of numeric and categorical variables
    - Target: 7 personality categories (0-6)
    """)
    
    print("\n2. MISSING VALUE HANDLING")
    print("-" * 50)
    print("""
    Missing values are handled using different strategies:
    
    a) Numeric Features:
       - Strategy: Median imputation
       - Reason: Median is robust to outliers
       - Implementation: SimpleImputer(strategy='median')
    
    b) Categorical Features:
       - Strategy: Mode (most frequent) imputation
       - Reason: Preserves the most common category
       - Implementation: SimpleImputer(strategy='most_frequent')
    
    Code implementation:
    num_imputer = SimpleImputer(strategy='median')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    """)
    
    print("\n3. FEATURE ENCODING STRATEGY")
    print("-" * 50)
    print("""
    Two encoding strategies are used based on cardinality:
    
    a) Low Cardinality (≤ 5 unique values):
       - Method: One-Hot Encoding
       - Reason: Preserves categorical nature, no ordinal relationship
       - Implementation: OneHotEncoder with handle_unknown='ignore'
    
    b) High Cardinality (> 5 unique values):
       - Method: Target Encoding
       - Reason: Reduces dimensionality, captures target relationship
       - Implementation: TargetEncoder from category_encoders
    
    Code implementation:
    for col in cat_cols:
        if X[col].nunique() <= 5:
            # One-hot encoding
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(X[[col]])
            ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            # ... implementation details
        else:
            # Target encoding
            te = TargetEncoder()
            te.fit(X[col], y)
            X[col] = te.transform(X[col])
    """)
    
    print("\n4. FEATURE SCALING")
    print("-" * 50)
    print("""
    StandardScaler is used for numeric features:
    
    - Method: StandardScaler (Z-score normalization)
    - Formula: (x - μ) / σ
    - Reason: Ensures all features contribute equally to model performance
    - Implementation: StandardScaler().fit_transform()
    
    Code implementation:
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    """)
    
    print("\n5. DATA BALANCING CONSIDERATIONS")
    print("-" * 50)
    print("""
    The dataset shows relatively balanced class distribution:
    
    - Imbalance ratio: 1.02 (very balanced)
    - All classes have similar representation (14.2% ± 0.5%)
    - No aggressive balancing techniques needed
    - Stratified sampling used in train-test split
    
    Implementation:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    """)

def document_model_creation():
    """Document the model creation and training process"""
    print("\n" + "=" * 60)
    print("MODEL CREATION AND TRAINING IMPLEMENTATION")
    print("=" * 60)
    
    print("\n1. MODEL SELECTION STRATEGY")
    print("-" * 50)
    print("""
    Seven different models are evaluated to ensure comprehensive comparison:
    
    a) Linear Models:
       - LogisticRegression: Linear baseline model
       - Use case: Simple, interpretable baseline
    
    b) Tree-Based Models:
       - DecisionTreeClassifier: Non-linear tree model
       - RandomForestClassifier: Ensemble tree model
       - Use case: Captures non-linear relationships
    
    c) Distance-Based Models:
       - KNeighborsClassifier: Distance-based classifier
       - Use case: Non-parametric, local learning
    
    d) Boosting Models:
       - XGBClassifier: Gradient boosting
       - LGBMClassifier: Light gradient boosting
       - CatBoostClassifier: Categorical boosting
       - Use case: High performance, handles complex patterns
    
    Implementation:
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=42),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(n_jobs=-1, random_state=42, verbose=0),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
    """)
    
    print("\n2. HYPERPARAMETER TUNING APPROACH")
    print("-" * 50)
    print("""
    Two-level hyperparameter tuning strategy:
    
    a) Basic Tuning (GridSearchCV):
       - Models: LogisticRegression, DecisionTree, RandomForest, KNN
       - Method: GridSearchCV with 5-fold cross-validation
       - Reason: Efficient for smaller parameter spaces
    
    b) Advanced Tuning (Optuna):
       - Models: XGBoost, LightGBM, CatBoost
       - Method: Optuna optimization with 50 trials
       - Reason: Large parameter spaces, efficient search
    
    Implementation:
    # Basic tuning
    gs = GridSearchCV(model, param_grids[name], cv=skf, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    
    # Advanced tuning (Optuna)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    """)
    
    print("\n3. CROSS-VALIDATION STRATEGY")
    print("-" * 50)
    print("""
    Stratified K-Fold Cross-Validation:
    
    - Folds: 5 (k=5)
    - Method: StratifiedKFold
    - Reason: Maintains class distribution in each fold
    - Random state: 42 (reproducibility)
    
    Implementation:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    Benefits:
    - Robust performance estimation
    - Handles class imbalance
    - Reduces overfitting risk
    """)
    
    print("\n4. EVALUATION METRICS")
    print("-" * 50)
    print("""
    Comprehensive evaluation using multiple metrics:
    
    a) Primary Metrics:
       - Accuracy: Overall correct predictions
       - F1-Score: Harmonic mean of precision and recall
       - Precision: True positives / (True positives + False positives)
       - Recall: True positives / (True positives + False negatives)
    
    b) Secondary Metrics:
       - ROC-AUC: Area under ROC curve (where applicable)
       - Cross-validation scores: Mean and standard deviation
    
    Implementation:
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    """)

def document_prediction_and_submission():
    """Document the prediction and submission process"""
    print("\n" + "=" * 60)
    print("PREDICTION AND SUBMISSION IMPLEMENTATION")
    print("=" * 60)
    
    print("\n1. FEATURE ALIGNMENT")
    print("-" * 50)
    print("""
    Critical step: Ensure test data has same features as training data
    
    Process:
    1. Identify missing features in test set
    2. Add missing features with default values (0)
    3. Remove extra features from test set
    4. Ensure same column order as training
    
    Implementation:
    # Find missing features
    missing_features = set(train_features) - set(test_features)
    for feature in missing_features:
        X_test[feature] = 0
    
    # Remove extra features
    extra_features = set(test_features) - set(train_features)
    X_test = X_test.drop(columns=list(extra_features))
    
    # Ensure same order
    X_test = X_test[train_features]
    """)
    
    print("\n2. PREDICTION PROCESS")
    print("-" * 50)
    print("""
    Step-by-step prediction process:
    
    1. Load preprocessed test data
    2. Apply same preprocessing pipeline
    3. Make predictions using best model
    4. Decode predictions back to original labels
    5. Create submission file
    
    Implementation:
    # Load test data
    X_test = pd.read_csv('test_preprocessed.csv')
    
    # Make predictions
    y_test_pred = best_model.predict(X_test)
    
    # Decode predictions
    y_test_pred_label = [y_inv_map.get(i, i) for i in y_test_pred]
    
    # Create submission
    sub = pd.DataFrame({'id': test['id'], 'Personality': y_test_pred_label})
    sub.to_csv('submission.csv', index=False)
    """)
    
    print("\n3. MODEL PERSISTENCE")
    print("-" * 50)
    print("""
    Save trained model and preprocessing objects for future use:
    
    Objects saved:
    - best_model.pkl: Trained model
    - num_imputer.pkl: Numeric imputer
    - cat_imputer.pkl: Categorical imputer
    - encoders.pkl: Feature encoders
    - scaler.pkl: Feature scaler
    - y_inv_map.pkl: Target encoding mapping
    
    Implementation:
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(num_imputer, 'num_imputer.pkl')
    joblib.dump(cat_imputer, 'cat_imputer.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(y_inv_map, 'y_inv_map.pkl')
    """)
    
    print("\n4. SUBMISSION FILE FORMAT")
    print("-" * 50)
    print("""
    Submission file requirements:
    
    - Format: CSV
    - Columns: id, Personality
    - Index: False (no row numbers)
    - Encoding: UTF-8
    
    Example submission format:
    id,Personality
    1,Analyst
    2,Diplomat
    3,Sentinel
    ...
    
    Implementation:
    sub = pd.DataFrame({
        'id': test['id'], 
        'Personality': y_test_pred_label
    })
    sub.to_csv('submission.csv', index=False)
    """)

def demonstrate_practical_implementation():
    """Demonstrate practical implementation with code examples"""
    print("\n" + "=" * 60)
    print("PRACTICAL IMPLEMENTATION DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. COMPLETE PIPELINE EXAMPLE")
    print("-" * 50)
    
    # Load data
    print("Loading data...")
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        print(f"✓ Data loaded: Train shape {train.shape}, Test shape {test.shape}")
    except FileNotFoundError:
        print("⚠ Data files not found. Using sample data for demonstration.")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        train = pd.DataFrame({
            'id': range(n_samples),
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
            'Personality': np.random.choice(['Analyst', 'Diplomat', 'Sentinel'], n_samples)
        })
        test = pd.DataFrame({
            'id': range(500),
            'feature1': np.random.normal(0, 1, 500),
            'feature2': np.random.normal(0, 1, 500),
            'categorical': np.random.choice(['A', 'B', 'C'], 500)
        })
    
    # Separate target
    y = train['Personality']
    X = train.drop(['Personality'], axis=1)
    X_test_orig = test.copy()
    
    print("✓ Target separated")
    
    # Preprocessing
    print("\nPreprocessing data...")
    
    # Drop ID column
    if 'id' in X.columns:
        X = X.drop('id', axis=1)
    if 'id' in X_test_orig.columns:
        X_test_orig = X_test_orig.drop('id', axis=1)
    
    # Identify columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"✓ Features identified: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    
    # Imputation
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        X_test = X_test_orig.copy()
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        print("✓ Numeric imputation completed")
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        print("✓ Categorical imputation completed")
    
    # Encoding
    encoders = {}
    if len(cat_cols) > 0:
        for col in cat_cols:
            if X[col].nunique() <= 5:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                ohe.fit(X[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                ohe_train = pd.DataFrame(ohe.transform(X[[col]]), columns=ohe_cols, index=X.index)
                ohe_test = pd.DataFrame(ohe.transform(X_test[[col]]), columns=ohe_cols, index=X_test.index)
                
                X = pd.concat([X.drop(col, axis=1), ohe_train], axis=1)
                X_test = pd.concat([X_test.drop(col, axis=1), ohe_test], axis=1)
                encoders[col] = ohe
                print(f"✓ {col}: One-hot encoded -> {len(ohe_cols)} features")
            else:
                te = TargetEncoder()
                te.fit(X[col], y)
                X[col] = te.transform(X[col])
                X_test[col] = te.transform(X_test[col])
                encoders[col] = te
                print(f"✓ {col}: Target encoded")
    
    # Scaling
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        print("✓ Feature scaling completed")
    
    print("✓ Preprocessing completed")
    
    # Model training
    print("\nTraining model...")
    
    # Encode target
    y_map = {k: v for v, k in enumerate(y.unique())}
    y_inv_map = {v: k for k, v in y_map.items()}
    y_enc = y.map(y_map)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Train Random Forest (best model from evaluation)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_val_pred = rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"✓ Model trained: Accuracy={accuracy:.4f}, F1-Score={f1:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_test_pred = rf_model.predict(X_test)
    y_test_pred_label = [y_inv_map.get(i, i) for i in y_test_pred]
    
    # Create submission
    sub = pd.DataFrame({'id': test['id'], 'Personality': y_test_pred_label})
    sub.to_csv('sample_submission.csv', index=False)
    
    print("✓ Predictions saved to sample_submission.csv")
    
    # Display results
    print(f"\nPREDICTION STATISTICS")
    print("-" * 40)
    print(f"Total predictions: {len(y_test_pred_label)}")
    pred_dist = pd.Series(y_test_pred_label).value_counts()
    for personality, count in pred_dist.items():
        print(f"  {personality}: {count} ({count/len(y_test_pred_label)*100:.1f}%)")
    
    print(f"\nFIRST 10 PREDICTIONS")
    print("-" * 40)
    print(sub.head(10))
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION DEMONSTRATION COMPLETED!")
    print("=" * 60)

def main():
    """Main function to run implementation documentation"""
    print("=" * 60)
    print("IMPLEMENTATION DETAILS FOR PERSONALITY CLASSIFICATION")
    print("=" * 60)
    
    # Document each section
    document_data_preprocessing()
    document_model_creation()
    document_prediction_and_submission()
    demonstrate_practical_implementation()
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION DOCUMENTATION COMPLETED!")
    print("=" * 60)
    print("\nKey Implementation Points:")
    print("1. Robust preprocessing pipeline with proper feature alignment")
    print("2. Comprehensive model evaluation with multiple algorithms")
    print("3. Advanced hyperparameter tuning using Optuna")
    print("4. Proper model persistence and deployment strategy")
    print("5. Thorough error analysis and performance comparison")

if __name__ == "__main__":
    main() 