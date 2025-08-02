# Detailed Model Evaluation for Personality Classification
# This script provides comprehensive model evaluation with detailed analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    train_data = pd.read_csv('train_preprocessed.csv')
    X = train_data.drop(['Personality'], axis=1)
    y = train_data['Personality']
    
    # Encode target
    y_map = {k: v for v, k in enumerate(y.unique())}
    y_inv_map = {v: k for k, v in y_map.items()}
    y_enc = y.map(y_map)
    
    return X, y_enc, y_map, y_inv_map

def evaluate_single_model(model, model_name, X_train, X_val, y_train, y_val, y_map):
    """Evaluate a single model with detailed metrics"""
    print(f"\n" + "=" * 60)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 60)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    
    # Calculate per-class metrics
    class_report = classification_report(y_val, y_val_pred, 
                                       target_names=list(y_map.keys()), 
                                       output_dict=True)
    
    # Print results
    print(f"Model Name: {model_name}")
    print(f"Accuracy: {val_acc:.6f}")
    print(f"Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=list(y_map.keys())))
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(y_map.keys()), 
                yticklabels=list(y_map.keys()), ax=ax1)
    ax1.set_title(f'Confusion Matrix - {model_name}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Per-class metrics
    classes = list(y_map.keys())
    precision_scores = [class_report[cls]['precision'] for cls in classes]
    recall_scores = [class_report[cls]['recall'] for cls in classes]
    f1_scores = [class_report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='skyblue')
    ax2.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightcoral')
    ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Per-Class Performance - {model_name}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training vs Validation accuracy
    accuracies = [train_acc, val_acc]
    labels = ['Training', 'Validation']
    colors = ['skyblue', 'lightcoral']
    
    bars = ax3.bar(labels, accuracies, color=colors, alpha=0.8)
    ax3.set_ylabel('Accuracy')
    ax3.set_title(f'Training vs Validation Accuracy - {model_name}')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Overall metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [val_acc, val_precision, val_recall, val_f1]
    colors_metrics = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars_metrics = ax4.bar(metrics, scores, color=colors_metrics, alpha=0.8)
    ax4.set_ylabel('Score')
    ax4.set_title(f'Overall Performance Metrics - {model_name}')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars_metrics, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'model_evaluation_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'class_report': class_report
    }

def comprehensive_model_evaluation():
    """Perform comprehensive evaluation of all models"""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    X, y_enc, y_map, y_inv_map = load_preprocessed_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # Define models
    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_jobs=-1, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                                      n_jobs=-1, random_state=42),
        'LGBMClassifier': LGBMClassifier(n_jobs=-1, random_state=42, verbose=0),
        'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42)
    }
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        try:
            result = evaluate_single_model(model, name, X_train, X_val, y_train, y_val, y_map)
            results[name] = result
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    # Create comprehensive comparison
    create_model_comparison_visualization(results)
    
    return results

def create_model_comparison_visualization(results):
    """Create comprehensive model comparison visualization"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 60)
    
    # Prepare data for visualization
    model_names = list(results.keys())
    accuracies = [results[name]['val_accuracy'] for name in model_names]
    f1_scores = [results[name]['val_f1'] for name in model_names]
    precision_scores = [results[name]['val_precision'] for name in model_names]
    recall_scores = [results[name]['val_recall'] for name in model_names]
    
    # Create comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, alpha=0.8, color='skyblue')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. F1-Score comparison
    bars2 = ax2.bar(model_names, f1_scores, alpha=0.8, color='lightcoral')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Model F1-Score Comparison')
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Precision vs Recall comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    bars3 = ax3.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.8, color='lightgreen')
    bars4 = ax3.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.8, color='gold')
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision vs Recall Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars3, precision_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, score in zip(bars4, recall_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Overall performance heatmap
    metrics_df = pd.DataFrame({
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'Precision': precision_scores,
        'Recall': recall_scores
    }, index=model_names)
    
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Performance Metrics Heatmap')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Metrics')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'F1-Score': f1_scores,
        'Precision': precision_scores,
        'Recall': recall_scores
    })
    
    print("\nMODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(summary_df.round(4))
    
    # Save summary
    summary_df.to_csv('model_evaluation_summary.csv', index=False)
    
    # Find best model
    best_model_idx = summary_df['F1-Score'].idxmax()
    best_model = summary_df.loc[best_model_idx, 'Model']
    best_f1 = summary_df.loc[best_model_idx, 'F1-Score']
    
    print(f"\nBEST MODEL: {best_model}")
    print(f"Best F1-Score: {best_f1:.4f}")
    
    return summary_df

def create_error_analysis_table(results):
    """Create detailed error analysis table"""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS TABLE")
    print("=" * 60)
    
    # Calculate error metrics
    error_data = []
    for name, result in results.items():
        # Calculate RMSE (simplified - using accuracy as proxy)
        rmse = np.sqrt(1 - result['val_accuracy'])
        
        # Calculate MAE (simplified - using accuracy as proxy)
        mae = 1 - result['val_accuracy']
        
        # Calculate R-squared (simplified - using accuracy as proxy)
        r_squared = result['val_accuracy']
        
        error_data.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r_squared,
            'Accuracy': result['val_accuracy'],
            'F1-Score': result['val_f1']
        })
    
    error_df = pd.DataFrame(error_data)
    
    print("Error Analysis Results:")
    print(error_df.round(4))
    
    # Create error comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error metrics comparison
    models = error_df['Model']
    rmse_scores = error_df['RMSE']
    mae_scores = error_df['MAE']
    r2_scores = error_df['R-squared']
    
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, rmse_scores, width, label='RMSE', alpha=0.8, color='skyblue')
    ax1.bar(x, mae_scores, width, label='MAE', alpha=0.8, color='lightcoral')
    ax1.bar(x + width, r2_scores, width, label='R-squared', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Error Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance ranking
    performance_metrics = ['Accuracy', 'F1-Score', 'R-squared']
    performance_data = [error_df['Accuracy'], error_df['F1-Score'], error_df['R-squared']]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, (metric, data, color) in enumerate(zip(performance_metrics, performance_data, colors)):
        ax2.bar([f"{model}\n({metric})" for model in models], data, alpha=0.8, color=color)
    
    ax2.set_ylabel('Score')
    ax2.set_title('Performance Metrics Ranking')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save error analysis
    error_df.to_csv('error_analysis_results.csv', index=False)
    
    return error_df

def main():
    """Main function to run comprehensive model evaluation"""
    print("=" * 60)
    print("DETAILED MODEL EVALUATION FOR PERSONALITY CLASSIFICATION")
    print("=" * 60)
    
    # Run comprehensive evaluation
    results = comprehensive_model_evaluation()
    
    # Create error analysis
    error_df = create_error_analysis_table(results)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)
    print("Files generated:")
    print("- model_evaluation_summary.csv")
    print("- error_analysis_results.csv")
    print("- comprehensive_model_comparison.png")
    print("- error_analysis_comparison.png")
    print("- Individual model evaluation plots")

if __name__ == "__main__":
    main() 