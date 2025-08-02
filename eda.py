import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create output dir
os.makedirs('eda_output', exist_ok=True)

# 1. Data Overview
with open('eda_output/summary.txt', 'w') as f:
    f.write('Train shape: %s\n' % (train.shape,))
    f.write('Test shape: %s\n' % (test.shape,))
    f.write('\nTrain info:\n')
    train.info(buf=f)
    f.write('\nTest info:\n')
    test.info(buf=f)
    f.write('\nMissing values (train):\n')
    f.write(str(train.isnull().sum()))
    f.write('\n\nMissing values (test):\n')
    f.write(str(test.isnull().sum()))
    f.write('\n\nDuplicates in train: %d\n' % train.duplicated().sum())
    f.write('Duplicates in test: %d\n' % test.duplicated().sum())
    f.write('\nConstant features (train): %s\n' % list(train.columns[train.nunique()==1]))
    f.write('\nCardinality (train):\n')
    for col in train.columns:
        f.write(f'{col}: {train[col].nunique()}\n')
    f.write('\nClass distribution (train):\n')
    if 'Personality' in train.columns:
        f.write(str(train['Personality'].value_counts()))

# 2. Visualizations
# Numeric distributions
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    plt.figure()
    sns.histplot(train[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'eda_output/hist_{col}.png')
    plt.close()

# Categorical distributions
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    plt.figure()
    train[col].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'Value counts of {col}')
    plt.savefig(f'eda_output/bar_{col}.png')
    plt.close()

# Correlation heatmap
if len(num_cols) > 1:
    plt.figure(figsize=(10,8))
    corr = train[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('eda_output/corr_heatmap.png')
    plt.close()

# Pairplot for numeric features (sampled for speed)
if len(num_cols) > 1:
    sns.pairplot(train.sample(min(500, len(train))), vars=num_cols, hue='Personality' if 'Personality' in train.columns else None)
    plt.savefig('eda_output/pairplot.png')
    plt.close() 