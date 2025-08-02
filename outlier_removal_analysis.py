# Outlier Removal Analysis for Personality Classification
# This script demonstrates comprehensive outlier detection and removal techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for outlier analysis"""
    print("Loading data for outlier analysis...")
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Separate target
    y = train['Personality']
    X = train.drop(['Personality'], axis=1)
    
    # Drop ID column if present
    if 'id' in X.columns:
        X = X.drop('id', axis=1)
    
    return X, y

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create mask for outliers
    outlier_mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
    return data[outlier_mask], outlier_mask, (lower_bound, upper_bound)

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    outlier_mask = z_scores < threshold
    return data[outlier_mask], outlier_mask, z_scores

def analyze_outliers_for_feature(X, feature_name):
    """Comprehensive outlier analysis for a single feature"""
    print(f"\n" + "=" * 60)
    print(f"OUTLIER ANALYSIS FOR FEATURE: {feature_name}")
    print("=" * 60)
    
    # Get feature data
    feature_data = X[feature_name].dropna()
    
    # Basic statistics
    print(f"Feature Statistics:")
    print(f"  Mean: {feature_data.mean():.4f}")
    print(f"  Median: {feature_data.median():.4f}")
    print(f"  Std: {feature_data.std():.4f}")
    print(f"  Min: {feature_data.min():.4f}")
    print(f"  Max: {feature_data.max():.4f}")
    
    # IQR method
    X_iqr, iqr_mask, iqr_bounds = detect_outliers_iqr(X, feature_name)
    outliers_iqr = (~iqr_mask).sum()
    outlier_percentage_iqr = (outliers_iqr / len(X)) * 100
    
    print(f"\nIQR Method Results:")
    print(f"  Lower bound: {iqr_bounds[0]:.4f}")
    print(f"  Upper bound: {iqr_bounds[1]:.4f}")
    print(f"  Outliers detected: {outliers_iqr} ({outlier_percentage_iqr:.2f}%)")
    
    # Z-score method
    X_zscore, zscore_mask, z_scores = detect_outliers_zscore(X, feature_name)
    outliers_zscore = (~zscore_mask).sum()
    outlier_percentage_zscore = (outliers_zscore / len(X)) * 100
    
    print(f"\nZ-Score Method Results:")
    print(f"  Outliers detected: {outliers_zscore} ({outlier_percentage_zscore:.2f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original distribution
    axes[0, 0].hist(feature_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title(f'Original Distribution - {feature_name}')
    axes[0, 0].set_xlabel(feature_name)
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(feature_data, vert=False)
    axes[0, 1].set_title(f'Box Plot - {feature_name}')
    axes[0, 1].set_xlabel(feature_name)
    
    # IQR filtered distribution
    axes[1, 0].hist(X_iqr[feature_name], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title(f'Distribution After IQR Outlier Removal')
    axes[1, 0].set_xlabel(feature_name)
    axes[1, 0].set_ylabel('Frequency')
    
    # Z-score filtered distribution
    axes[1, 1].hist(X_zscore[feature_name], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_title(f'Distribution After Z-Score Outlier Removal')
    axes[1, 1].set_xlabel(feature_name)
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'eda_output/outlier_analysis_{feature_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'feature': feature_name,
        'iqr_outliers': outliers_iqr,
        'iqr_percentage': outlier_percentage_iqr,
        'zscore_outliers': outliers_zscore,
        'zscore_percentage': outlier_percentage_zscore,
        'total_samples': len(X)
    }

def comprehensive_outlier_analysis(X):
    """Perform comprehensive outlier analysis on all numeric features"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE OUTLIER ANALYSIS")
    print("=" * 60)
    
    # Get numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Analyzing {len(num_cols)} numeric features for outliers...")
    
    # Analyze each feature
    outlier_results = []
    for feature in num_cols:
        try:
            result = analyze_outliers_for_feature(X, feature)
            outlier_results.append(result)
        except Exception as e:
            print(f"Error analyzing {feature}: {e}")
    
    # Create summary
    summary_df = pd.DataFrame(outlier_results)
    
    print(f"\n" + "=" * 60)
    print("OUTLIER ANALYSIS SUMMARY")
    print("=" * 60)
    print(summary_df.round(2))
    
    # Save summary
    summary_df.to_csv('eda_output/outlier_analysis_summary.csv', index=False)
    
    # Create summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # IQR outliers percentage
    features = summary_df['feature']
    iqr_percentages = summary_df['iqr_percentage']
    zscore_percentages = summary_df['zscore_percentage']
    
    x = np.arange(len(features))
    width = 0.35
    
    ax1.bar(x - width/2, iqr_percentages, width, label='IQR Method', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, zscore_percentages, width, label='Z-Score Method', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Outlier Percentage (%)')
    ax1.set_title('Outlier Detection Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (iqr_pct, zscore_pct) in enumerate(zip(iqr_percentages, zscore_percentages)):
        ax1.text(i - width/2, iqr_pct + 0.1, f'{iqr_pct:.1f}%', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, zscore_pct + 0.1, f'{zscore_pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Total outliers comparison
    total_iqr = summary_df['iqr_outliers'].sum()
    total_zscore = summary_df['zscore_outliers'].sum()
    total_samples = summary_df['total_samples'].iloc[0]
    
    methods = ['IQR Method', 'Z-Score Method']
    total_outliers = [total_iqr, total_zscore]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax2.bar(methods, total_outliers, color=colors, alpha=0.8)
    ax2.set_ylabel('Total Outliers Detected')
    ax2.set_title('Total Outliers by Detection Method')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, total_outliers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_output/outlier_summary_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

def demonstrate_outlier_removal_impact(X, y, feature_name):
    """Demonstrate the impact of outlier removal on model performance"""
    print(f"\n" + "=" * 60)
    print(f"OUTLIER REMOVAL IMPACT ANALYSIS - {feature_name}")
    print("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    # Prepare data
    X_clean = X.dropna()
    y_clean = y[X_clean.index]
    
    # Original data performance
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model on original data
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train_scaled, y_train)
    y_pred_original = rf_original.predict(X_test_scaled)
    
    acc_original = accuracy_score(y_test, y_pred_original)
    f1_original = f1_score(y_test, y_pred_original, average='weighted')
    
    # Remove outliers using IQR
    X_iqr, iqr_mask, _ = detect_outliers_iqr(X_clean, feature_name)
    y_iqr = y_clean[iqr_mask]
    
    # Train model on outlier-removed data
    X_train_iqr, X_test_iqr, y_train_iqr, y_test_iqr = train_test_split(
        X_iqr, y_iqr, test_size=0.2, random_state=42, stratify=y_iqr
    )
    
    scaler_iqr = StandardScaler()
    X_train_iqr_scaled = scaler_iqr.fit_transform(X_train_iqr)
    X_test_iqr_scaled = scaler_iqr.transform(X_test_iqr)
    
    rf_iqr = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_iqr.fit(X_train_iqr_scaled, y_train_iqr)
    y_pred_iqr = rf_iqr.predict(X_test_iqr_scaled)
    
    acc_iqr = accuracy_score(y_test_iqr, y_pred_iqr)
    f1_iqr = f1_score(y_test_iqr, y_pred_iqr, average='weighted')
    
    # Create comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    methods = ['Original Data', 'IQR Outlier Removed']
    accuracies = [acc_original, acc_iqr]
    f1_scores = [f1_original, f1_iqr]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Data Treatment')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Model Performance: Original vs Outlier-Removed Data')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom')
    
    for bar, score in zip(bars2, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom')
    
    # Data size comparison
    sizes = [len(X_clean), len(X_iqr)]
    bars3 = ax2.bar(methods, sizes, alpha=0.8, color=['skyblue', 'lightgreen'])
    ax2.set_xlabel('Data Treatment')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Dataset Size Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, size in zip(bars3, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 f'{size}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'eda_output/outlier_impact_{feature_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results
    print(f"Original Data Performance:")
    print(f"  Accuracy: {acc_original:.4f}")
    print(f"  F1-Score: {f1_original:.4f}")
    print(f"  Samples: {len(X_clean)}")
    
    print(f"\nIQR Outlier-Removed Data Performance:")
    print(f"  Accuracy: {acc_iqr:.4f}")
    print(f"  F1-Score: {f1_iqr:.4f}")
    print(f"  Samples: {len(X_iqr)}")
    print(f"  Removed: {len(X_clean) - len(X_iqr)} samples ({((len(X_clean) - len(X_iqr))/len(X_clean)*100):.2f}%)")
    
    return {
        'original_accuracy': acc_original,
        'original_f1': f1_original,
        'iqr_accuracy': acc_iqr,
        'iqr_f1': f1_iqr,
        'samples_removed': len(X_clean) - len(X_iqr),
        'removal_percentage': ((len(X_clean) - len(X_iqr))/len(X_clean)*100)
    }

def main():
    """Main function to run comprehensive outlier analysis"""
    print("=" * 60)
    print("OUTLIER REMOVAL ANALYSIS FOR PERSONALITY CLASSIFICATION")
    print("=" * 60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Create output directory
    import os
    if not os.path.exists('eda_output'):
        os.makedirs('eda_output')
    
    # Comprehensive outlier analysis
    summary_df = comprehensive_outlier_analysis(X)
    
    # Demonstrate impact on a specific feature
    if len(X.select_dtypes(include=[np.number]).columns) > 0:
        feature_to_analyze = X.select_dtypes(include=[np.number]).columns[0]
        impact_results = demonstrate_outlier_removal_impact(X, y, feature_to_analyze)
        
        # Save impact results
        impact_df = pd.DataFrame([impact_results])
        impact_df.to_csv('eda_output/outlier_impact_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS COMPLETED!")
    print("=" * 60)
    print("Files generated:")
    print("- eda_output/outlier_analysis_summary.csv")
    print("- eda_output/outlier_summary_comparison.png")
    print("- eda_output/outlier_impact_results.csv")
    print("- Individual feature outlier analysis plots")

if __name__ == "__main__":
    main() 