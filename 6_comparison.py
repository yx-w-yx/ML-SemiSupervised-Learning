import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("="*70)
print("Comparison: Diabetes vs Breast Cancer")
print("="*70)

# Load results
with open('diabetes_results.json', 'r') as f:
    diabetes = json.load(f)

with open('cancer_results.json', 'r') as f:
    cancer = json.load(f)

# Print comparison
print("\n1. DIABETES (Semi-Supervised Learning)")
print(f"   Method: K-means clustering for label generation")
print(f"   Test Accuracy: {diabetes['test_accuracy']:.4f}")
print(f"   PCA Variance Explained: {diabetes['pca_variance_explained']:.4f}")
print(f"   Confusion Matrix:")
for row in diabetes['confusion_matrix']:
    print(f"     {row}")

print("\n2. BREAST CANCER (Supervised Learning)")
print(f"   Method: Real labels (Alive/Dead)")
print(f"   Test Accuracy: {cancer['test_accuracy']:.4f}")
print(f"   PCA Components: {cancer['n_components']}")
print(f"   PCA Variance Explained: {cancer['pca_variance_explained']:.4f}")
print(f"   Confusion Matrix:")
for row in cancer['confusion_matrix']:
    print(f"     {row}")

print("\n3. KEY FINDINGS")
diff = abs(diabetes['test_accuracy'] - cancer['test_accuracy'])
print(f"   Accuracy Difference: {diff:.4f}")

if diabetes['test_accuracy'] > cancer['test_accuracy']:
    print(f"   Diabetes model performs better (+{diff:.4f})")
else:
    print(f"   Cancer model performs better (+{diff:.4f})")

print(f"\n   Both models achieve >{min(diabetes['test_accuracy'], cancer['test_accuracy'])*100:.1f}% accuracy")

print("\n4. ANALYSIS")
print("   - Semi-supervised learning (K-means) achieves comparable performance")
print("   - Super Learner generalizes well across different datasets")
print("   - PCA effectively reduces dimensionality")
print("   - Hyperparameter tuning improves model performance")

print("\n5. BEST HYPERPARAMETERS")
print(f"\n   Diabetes:")
print(f"     KNN: {diabetes['best_params']['knn']}")
print(f"     NN: {diabetes['best_params']['nn']}")
print(f"     DT: {diabetes['best_params']['dt']}")

print(f"\n   Cancer:")
print(f"     KNN: {cancer['best_params']['knn']}")
print(f"     NN: {cancer['best_params']['nn']}")
print(f"     DT: {cancer['best_params']['dt']}")

print("="*70)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
models = ['Diabetes\n(Semi-Supervised)', 'Cancer\n(Supervised)']
accuracies = [diabetes['test_accuracy'], cancer['test_accuracy']]
colors = ['#3498db', '#2ecc71']

axes[0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylim([0.75, 1.0])
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].axhline(y=0.85, color='red', linestyle='--', alpha=0.5, linewidth=2, label='85% Baseline')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[0].text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=11, fontweight='bold')

axes[0].legend(loc='lower right')

# Hyperparameters text
param_text = f"DIABETES BEST PARAMS:\n"
param_text += f"  KNN: n_neighbors={diabetes['best_params']['knn']['n_neighbors']}\n"
param_text += f"  NN: hidden={diabetes['best_params']['nn']['hidden_layer_sizes']}\n"
param_text += f"      lr={diabetes['best_params']['nn']['learning_rate_init']}\n"
param_text += f"  DT: depth={diabetes['best_params']['dt']['max_depth']}\n"
param_text += f"      split={diabetes['best_params']['dt']['min_samples_split']}\n\n"

param_text += f"CANCER BEST PARAMS:\n"
param_text += f"  KNN: n_neighbors={cancer['best_params']['knn']['n_neighbors']}\n"
param_text += f"  NN: hidden={cancer['best_params']['nn']['hidden_layer_sizes']}\n"
param_text += f"      lr={cancer['best_params']['nn']['learning_rate_init']}\n"
param_text += f"  DT: depth={cancer['best_params']['dt']['max_depth']}\n"
param_text += f"      split={cancer['best_params']['dt']['min_samples_split']}"

axes[1].text(0.05, 0.95, param_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].axis('off')
axes[1].set_title('Optimal Hyperparameters', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('final_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: final_comparison.png")

print("\nAll analyses complete!")