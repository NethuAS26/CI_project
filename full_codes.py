import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
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
from category_encoders import TargetEncoder
warnings.filterwarnings('ignore')

class PersonalityClassifier:
    def __init__(self):
        self.num_imputer = None
        self.cat_imputer = None
        self.encoders = {}
        self.scaler = None
        self.best_model = None
        self.best_model_tuned = None
        self.y_map = None
        self.y_inv_map = None
        
    def preprocess_data(self, train_path='train.csv', test_path='test.csv'):
        """Preprocess training and test data"""
        print("Loading and preprocessing data...")
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        # Separate target
        y = train['Personality']
        X = train.drop(['Personality'], axis=1)
        X_test = test.copy()
        
        # Identify columns
        drop_cols = ['id']
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Impute numeric
        self.num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
        X_test[num_cols] = self.num_imputer.transform(X_test[num_cols])
        
        # Impute categorical
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])
        X_test[cat_cols] = self.cat_imputer.transform(X_test[cat_cols])
        
        # Encoding
        encoded_X = X.copy()
        encoded_X_test = X_test.copy()
        
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
                self.encoders[col] = ohe
            else:
                # Target encoding
                te = TargetEncoder()
                te.fit(X[col], y)
                encoded_X[col] = te.transform(X[col])
                encoded_X_test[col] = te.transform(X_test[col])
                self.encoders[col] = te
        
        # Scale numeric features
        self.scaler = StandardScaler()
        encoded_X[num_cols] = self.scaler.fit_transform(encoded_X[num_cols])
        encoded_X_test[num_cols] = self.scaler.transform(encoded_X_test[num_cols])
        
        # Save preprocessed data
        encoded_X['Personality'] = y
        encoded_X.to_csv('train_preprocessed.csv', index=False)
        encoded_X_test.to_csv('test_preprocessed.csv', index=False)
        
        # Save preprocessing objects
        joblib.dump(self.num_imputer, 'num_imputer.pkl')
        joblib.dump(self.cat_imputer, 'cat_imputer.pkl')
        joblib.dump(self.encoders, 'encoders.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        print("Preprocessing completed and saved!")
        return encoded_X, encoded_X_test
    
    def encode_target(self, y):
        """Encode target variable"""
        self.y_map = {k: v for v, k in enumerate(y.unique())}
        self.y_inv_map = {v: k for k, v in self.y_map.items()}
        return y.map(self.y_map)
    
    def train_models(self, X, y):
        """Train multiple models with basic hyperparameter tuning"""
        print("Training models with basic hyperparameter tuning...")
        
        y_enc = self.encode_target(y)
        
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.y_map.keys(), yticklabels=self.y_map.keys())
        plt.title(f'Confusion Matrix ({best_name})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Feature importance
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
            try:
                explainer = shap.Explainer(best_model, X)
                shap_values = explainer(X)
                shap.summary_plot(shap_values, X, show=False)
                plt.tight_layout()
                plt.savefig('feature_importance_shap.png')
                plt.close()
            except:
                print("Could not generate SHAP plot")
        
        # Save best model
        self.best_model = best_model
        joblib.dump(best_model, 'best_model.pkl')
        print(f'Best model: {best_name} (accuracy={best_score:.4f}) saved as best_model.pkl')
        
        return results, best_model, best_name
    
    def hyperparameter_tuning(self, X, y):
        """Advanced hyperparameter tuning using Optuna"""
        print("Starting comprehensive hyperparameter tuning...")
        
        y_enc = self.encode_target(y)
        
        # Define objective functions for each model
        def objective_logistic(trial):
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
            
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
                'p': trial.suggest_int('p', 1, 3)
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
        
        print("=" * 60)
        
        for model_name, objective_func in models_config.items():
            print(f"\nOptimizing {model_name}...")
            
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            study.optimize(objective_func, n_trials=50, show_progress_bar=True)
            
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
        self.best_model_tuned = best_model_instance
        joblib.dump(best_model_instance, f'best_model_hyperparameter_tuned.pkl')
        
        # Save inverse mapping
        joblib.dump(self.y_inv_map, 'y_inv_map.pkl')
        
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
        
        return results, best_model_instance, best_model_name
    
    def predict(self, X_test, model_type='tuned'):
        """Make predictions using the trained model"""
        if model_type == 'tuned' and self.best_model_tuned is not None:
            model = self.best_model_tuned
        elif self.best_model is not None:
            model = self.best_model
        else:
            raise ValueError("No trained model available")
        
        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            np.save('test_pred_proba.npy', y_pred_proba)
        
        y_pred = model.predict(X_test)
        
        # Decode predictions
        if self.y_inv_map is not None:
            y_pred_label = [self.y_inv_map.get(i, i) for i in y_pred]
        else:
            y_pred_label = y_pred
        
        return y_pred_label
    
    def create_submission(self, test_path='test.csv', model_type='tuned'):
        """Create submission file for Kaggle"""
        print(f"Creating submission using {model_type} model...")
        
        # Load test data
        X_test = pd.read_csv('test_preprocessed.csv')
        orig_test = pd.read_csv(test_path)
        ids = orig_test['id']
        
        # Make predictions
        y_pred_label = self.predict(X_test, model_type)
        
        # Create submission
        sub = pd.DataFrame({'id': ids, 'Personality': y_pred_label})
        filename = f'submission_{model_type}.csv'
        sub.to_csv(filename, index=False)
        
        print(f'Predictions saved to {filename}')
        print(f"\nFirst 10 predictions:")
        print(sub.head(10))
        print(f"\nPrediction distribution:")
        print(sub['Personality'].value_counts())
        
        return sub

def main():
    """Main function to run the complete pipeline"""
    print("=" * 60)
    print("PERSONALITY CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Initialize classifier
    classifier = PersonalityClassifier()
    
    # Step 1: Preprocess data
    train_data, test_data = classifier.preprocess_data()
    
    # Step 2: Train basic models
    print("\n" + "=" * 60)
    print("STEP 2: BASIC MODEL TRAINING")
    print("=" * 60)
    X = train_data.drop(['Personality'], axis=1)
    y = train_data['Personality']
    basic_results, basic_model, basic_name = classifier.train_models(X, y)
    
    # Step 3: Advanced hyperparameter tuning
    print("\n" + "=" * 60)
    print("STEP 3: ADVANCED HYPERPARAMETER TUNING")
    print("=" * 60)
    tuned_results, tuned_model, tuned_name = classifier.hyperparameter_tuning(X, y)
    
    # Step 4: Create submissions
    print("\n" + "=" * 60)
    print("STEP 4: CREATING SUBMISSIONS")
    print("=" * 60)
    
    # Basic model submission
    basic_sub = classifier.create_submission(model_type='basic')
    
    # Tuned model submission
    tuned_sub = classifier.create_submission(model_type='tuned')
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Files created:")
    print("- train_preprocessed.csv")
    print("- test_preprocessed.csv")
    print("- model_comparison.csv")
    print("- hyperparameter_tuning_results.csv")
    print("- best_model.pkl")
    print("- best_model_hyperparameter_tuned.pkl")
    print("- submission_basic.csv")
    print("- submission_tuned.csv")
    print("- Various visualization files (.png)")

if __name__ == "__main__":
    main() 