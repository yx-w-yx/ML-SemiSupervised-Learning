import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 5: Apply to Breast Cancer Dataset")
print("=" * 60)

# Read data
df = pd.read_csv('cancer_cleaned.csv')
print(f"Original shape: {df.shape}")

# Separate X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"\nLabel distribution:")
print(y.value_counts())
print(f"Label balance: {y.value_counts(normalize=True)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# PCA - determine optimal components
print("\nDetermining optimal PCA components...")
pca_temp = PCA()
pca_temp.fit(X_train)
cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
n_components = np.argmax(cumsum_var >= 0.90) + 1

print(f"Components for 90% variance: {n_components}")
print(f"Variance explained: {cumsum_var[n_components - 1]:.4f}")

# Apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"\nPCA shape: {X_train_pca.shape}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# ========================================
print("\n" + "=" * 60)
print("Step 4: Super Learner Classification")
print("=" * 60)

# Hyperparameter tuning
print("\nHyperparameter tuning...")

# KNN
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train_pca, y_train)
best_knn = knn_grid.best_estimator_
print(f"Best KNN: {knn_grid.best_params_}, CV score: {knn_grid.best_score_:.4f}")

# Neural Network
nn_params = {
    'hidden_layer_sizes': [(10,), (16,), (20,), (10, 10)],
    'learning_rate_init': [0.001, 0.01]
}
nn_grid = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, solver='adam'),
    nn_params, cv=5, scoring='accuracy'
)
nn_grid.fit(X_train_pca, y_train)
best_nn = nn_grid.best_estimator_
print(f"Best NN: {nn_grid.best_params_}, CV score: {nn_grid.best_score_:.4f}")

# Naive Bayes
nb = GaussianNB()

# Base models
base_models = [nb, best_knn, best_nn]
model_names = ['Naive Bayes', 'KNN', 'Neural Network']

# Generate meta features
print("\nGenerating meta features (5-fold CV)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((len(X_train_pca), len(base_models)))

for train_idx, val_idx in kf.split(X_train_pca):
    X_t, X_v = X_train_pca[train_idx], X_train_pca[val_idx]
    y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

    for i, model in enumerate(base_models):
        m = clone(model)
        m.fit(X_t, y_t)
        meta_train[val_idx, i] = m.predict_proba(X_v)[:, 1]

# Train meta learner
print("\nTraining meta learner (Decision Tree)...")
dt_params = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid.fit(meta_train, y_train)
meta_learner = dt_grid.best_estimator_
print(f"Best DT: {dt_grid.best_params_}, CV score: {dt_grid.best_score_:.4f}")

# Test set prediction
print("\nTesting on test set...")
trained_base = []
for model in base_models:
    m = clone(model)
    m.fit(X_train_pca, y_train)
    trained_base.append(m)

meta_test = np.zeros((len(X_test_pca), len(base_models)))
for i, model in enumerate(trained_base):
    meta_test[:, i] = model.predict_proba(X_test_pca)[:, 1]

y_pred = meta_learner.predict(meta_test)

# ========================================
print("\n" + "=" * 60)
print("Results")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Dead', 'Alive']))

# Base model performance
print("\nBase model accuracies:")
for model, name in zip(trained_base, model_names):
    base_pred = model.predict(X_test_pca)
    base_acc = accuracy_score(y_test, base_pred)
    print(f"  {name}: {base_acc:.4f}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Dead', 'Alive'],
            yticklabels=['Dead', 'Alive'])
plt.title(f'Cancer - Confusion Matrix (Accuracy: {accuracy:.3f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('cancer_confusion_matrix.png', dpi=300)
print("\nSaved: cancer_confusion_matrix.png")

# Save results
import json

results = {
    'test_accuracy': float(accuracy),
    'confusion_matrix': cm.tolist(),
    'n_components': int(n_components),
    'best_params': {
        'knn': knn_grid.best_params_,
        'nn': nn_grid.best_params_,
        'dt': dt_grid.best_params_
    },
    'pca_variance_explained': float(sum(pca.explained_variance_ratio_))
}

with open('cancer_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nCancer modeling complete!")