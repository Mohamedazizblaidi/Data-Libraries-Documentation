## Introduction

Scikit-learn is a powerful machine learning library for Python that provides simple and efficient tools for data mining and data analysis. It's built on NumPy, SciPy, and matplotlib and offers a consistent interface for various machine learning algorithms.

### Key Features

- Simple and efficient tools for predictive data analysis
- Accessible to everybody and reusable in various contexts
- Built on NumPy, SciPy, and matplotlib
- Open source, commercially usable (BSD license)
- Extensive documentation and examples

### Main Components

- **Supervised Learning**: Classification and regression
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Model Selection**: Cross-validation and hyperparameter tuning
- **Data Preprocessing**: Feature scaling, encoding, and transformation
- **Feature Selection**: Selecting relevant features
- **Metrics**: Evaluation metrics for model performance

---

## Installation

### Using pip

```bash
pip install scikit-learn
```

### Using conda

```bash
conda install scikit-learn
```

### With additional dependencies

```bash
pip install scikit-learn[alldeps]
```

### Basic Imports

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```

---

## Basic Concepts

### The Scikit-Learn API

All estimators in scikit-learn share a uniform interface consisting of:

#### **1. Estimator Interface**

```python
# All algorithms implement fit() and predict()
estimator = SomeAlgorithm(param1=value1, param2=value2)
estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)
```

#### **2. Transformer Interface**

```python
# For preprocessing and feature transformation
transformer = SomeTransformer(param1=value1)
transformer.fit(X_train)
X_transformed = transformer.transform(X_test)
# Or combined: X_transformed = transformer.fit_transform(X_train)
```

#### **3. Predictor Interface**

```python
# For making predictions
predictor.predict(X_test)          # Point predictions
predictor.predict_proba(X_test)    # Probability estimates
predictor.decision_function(X_test) # Decision scores
```

### Data Format

- **Features (X)**: 2D array-like, shape (n_samples, n_features)
- **Target (y)**: 1D array-like, shape (n_samples,)
- **Sample weight**: 1D array-like, shape (n_samples,) - optional

```python
# Example data format
X = np.array([[1, 2], [3, 4], [5, 6]])  # Features
y = np.array([0, 1, 0])                  # Target labels
```

---

## Data Preprocessing

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max scaling (0 to 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# Robust scaling (median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Label Encoding (for target variables)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X_categorical)

# Ordinal Encoding
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

### Handling Missing Values

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent', 'constant'
X_imputed = imputer.fit_transform(X)

# KNN imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

### Feature Generation

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

---

## Supervised Learning

### Classification

#### **Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)
```

#### **Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
feature_importance = clf.feature_importances_
```

#### **Support Vector Machine**

```python
from sklearn.svm import SVC

clf = SVC(kernel='rbf', probability=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

#### **Decision Tree**

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

#### **K-Nearest Neighbors**

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

#### **Naive Bayes**

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

#### **Gradient Boosting**

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### Regression

#### **Linear Regression**

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
coefficients = reg.coef_
intercept = reg.intercept_
```

#### **Ridge Regression**

```python
from sklearn.linear_model import Ridge

reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

#### **Lasso Regression**

```python
from sklearn.linear_model import Lasso

reg = Lasso(alpha=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

#### **Random Forest Regression**

```python
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

#### **Support Vector Regression**

```python
from sklearn.svm import SVR

reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

---

## Unsupervised Learning

### Clustering

#### **K-Means**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_
```

#### **Hierarchical Clustering**

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels = clustering.fit_predict(X)
```

#### **DBSCAN**

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X)
```

### Dimensionality Reduction

#### **Principal Component Analysis (PCA)**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
```

#### **t-SNE**

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)
```

#### **Linear Discriminant Analysis (LDA)**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

---

## Model Selection

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, KFold

# Simple cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Cross-validation with multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(clf, X, y, cv=5, scoring=scoring)

# Custom cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv)
```

### Hyperparameter Tuning

#### **Grid Search**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

#### **Random Search**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### Validation Curves

```python
from sklearn.model_selection import validation_curve

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range,
    cv=5, scoring='accuracy', n_jobs=-1
)
```

---

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Comprehensive report
report = classification_report(y_test, y_pred)
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
auc = roc_auc_score(y_test, y_proba[:, 1])
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)

# Basic metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

# Additional metrics
explained_var = explained_variance_score(y_test, y_pred)
median_ae = median_absolute_error(y_test, y_pred)
```

### Clustering Metrics

```python
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score
)

# External metrics (when true labels are available)
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)

# Internal metrics
silhouette = silhouette_score(X, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
```

---

## Feature Engineering

### Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, RFE, SelectFromModel
)

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = selector.get_support(indices=True)

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Model-based feature selection
selector = SelectFromModel(RandomForestClassifier())
X_selected = selector.fit_transform(X, y)
```

### Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Text feature extraction
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(text_data)

# Count vectorizer
vectorizer = CountVectorizer(max_features=1000)
X_counts = vectorizer.fit_transform(text_data)
```

---

## Pipeline and Automation

### Creating Pipelines

```python
from sklearn.pipeline import Pipeline, make_pipeline

# Method 1: Using Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Method 2: Using make_pipeline
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Column Transformer

```python
from sklearn.compose import ColumnTransformer

# Different preprocessing for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

### Feature Union

```python
from sklearn.pipeline import FeatureUnion

# Combine multiple feature extraction methods
feature_union = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('poly', PolynomialFeatures(degree=2))
])

pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', RandomForestClassifier())
])
```

---

## Advanced Topics

### Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

# Bagging
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

# AdaBoost
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
```

### Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Custom transformation logic
        return X  # Return transformed data
```

### Model Persistence

```python
import joblib
import pickle

# Save model
joblib.dump(model, 'model.pkl')
# or
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
model = joblib.load('model.pkl')
# or
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

---

## Function Reference

### Core Functions

```python
train_test_split(X, y, test_size=0.25, random_state=None, stratify=None)`
```

Splits data into train and test sets.

- `X`: Features
- `y`: Target
- `test_size`: Proportion of test set (0.0-1.0)
- `random_state`: Random seed for reproducibility
- `stratify`: Stratify split by target classes
- Returns: `X_train, X_test, y_train, y_test`

```python
cross_val_score(estimator, X, y, cv=None, scoring=None, n_jobs=None)`
```

Evaluates model using cross-validation.

- `estimator`: Model to evaluate
- `X`: Features
- `y`: Target
- `cv`: Cross-validation strategy (int or CV object)
- `scoring`: Scoring metric
- `n_jobs`: Number of parallel jobs
- Returns: Array of scores

```python
GridSearchCV(estimator, param_grid, cv=None, scoring=None, n_jobs=None)`
```

Exhaustive search over parameter values.

- `estimator`: Model to tune
- `param_grid`: Dictionary of parameters to search
- `cv`: Cross-validation strategy
- `scoring`: Scoring metric
- `n_jobs`: Number of parallel jobs

### Preprocessing Functions

```python
StandardScaler(copy=True, with_mean=True, with_std=True)`
```

Standardizes features by removing mean and scaling to unit variance.

- `copy`: Copy data or modify in-place
- `with_mean`: Center data before scaling
- `with_std`: Scale to unit variance

```python
OneHotEncoder(categories='auto', drop=None, sparse=True)`
```

Encodes categorical features as one-hot vectors.

- `categories`: Categories per feature
- `drop`: Drop one category to avoid multicollinearity
- `sparse`: Return sparse matrix

```python
SimpleImputer(missing_values=np.nan, strategy='mean')`
```

Imputes missing values with simple strategies.

- `missing_values`: Placeholder for missing values
- `strategy`: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')

### Model Classes

```python
RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None)`
```

Random Forest classifier.

- `n_estimators`: Number of trees
- `max_depth`: Maximum depth of trees
- `random_state`: Random seed

```python
LogisticRegression(penalty='l2', C=1.0, random_state=None, max_iter=100)`
```

Logistic regression classifier.

- `penalty`: Regularization penalty ('l1', 'l2', 'elasticnet')
- `C`: Regularization strength (inverse)
- `max_iter`: Maximum iterations

```python
SVC(C=1.0, kernel='rbf', degree=3, gamma='scale')`
```

Support Vector Machine classifier.

- `C`: Regularization parameter
- `kernel`: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
- `degree`: Degree of polynomial kernel
- `gamma`: Kernel coefficient

### Evaluation Functions

``` python
accuracy_score(y_true, y_pred, normalize=True)`
```

Calculates accuracy score.

- `y_true`: True labels
- `y_pred`: Predicted labels
- `normalize`: Return fraction or count

```python
classification_report(y_true, y_pred, labels=None, target_names=None)`
```

Builds text report of classification metrics.

- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: Labels to include
- `target_names`: Display names for labels

```python
confusion_matrix(y_true, y_pred, labels=None)`
```

Computes confusion matrix.

- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: Labels to include

---

## Best Practices

### Data Preparation

```python
# Always split data before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit transformers on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Don't fit on test data!
```

### Model Development

```python
# Use pipelines for reproducible workflows
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Always set random_state for reproducibility
model = RandomForestClassifier(random_state=42)
```

### Evaluation

```python
# Use cross-validation for model selection
scores = cross_val_score(model, X_train, y_train, cv=5)

# Use separate test set for final evaluation
model.fit(X_train, y_train)
final_score = model.score(X_test, y_test)
```

### Performance Optimization

```python
# Use n_jobs=-1 for parallel processing
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
```

---

## Common Issues and Solutions

### Issue: Data Leakage

```python
# WRONG: Scaling before splitting
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# CORRECT: Split first, then scale
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Issue: Overfitting

```python
# Solutions:
# 1. Use cross-validation
scores = cross_val_score(model, X, y, cv=5)

# 2. Regularization
model = LogisticRegression(C=0.1)  # Stronger regularization

# 3. Feature selection
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)

# 4. More data or simpler model
model = RandomForestClassifier(max_depth=5)
```

### Issue: Imbalanced Classes

```python
# Solutions:
# 1. Stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# 2. Class weights
model = RandomForestClassifier(class_weight='balanced')

# 3. Resampling (using imbalanced-learn)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Issue: Memory Issues

```python
# Solutions:
# 1. Batch processing
from sklearn.utils import gen_batches
for batch in gen_batches(len(X), batch_size=1000):
    X_batch = X[batch]
    # Process batch

# 2. Use sparse matrices
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)

# 3. Feature selection
selector = SelectKBest(k=100)
X_reduced = selector.fit_transform(X, y)
```

### Issue: Categorical Variables

```python
# For tree-based models: Use OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X_categorical)

# For linear models: Use OneHotEncoder
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X_categorical)
```

---

## Quick Reference

### Common Estimators

- **Classification**: `LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`
- **Regression**: `LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`
- **Clustering**: `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- **Dimensionality Reduction**: `PCA`, `TSNE`, `LinearDiscriminantAnalysis`

### Common Transformers

- **Scaling**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- **Encoding**: `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`
- **Imputation**: `SimpleImputer`, `KNNImputer`
- **Feature Selection**: `SelectKBest`, `RFE`, `SelectFromModel`

### Common Metrics

- **Classification**: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
- **Regression**: `mean_squared_error`, `mean_absolute_error`, `r2_score`
- **Clustering**: `silhouette_score`, `adjusted_rand_score`

### Workflow Template

```python
# 1. Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 3. Hyperparameter tuning
param_grid = {'classifier__n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 4. Final evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

_This documentation provides a comprehensive overview of scikit-learn for machine learning tasks. For more advanced features and detailed examples, refer to the official scikit-learn documentation._
