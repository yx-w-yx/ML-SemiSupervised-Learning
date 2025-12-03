import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Read cleaned data
df = pd.read_csv('diabetes_cleaned.csv')
print(f"Original shape: {df.shape}")

# Select features for clustering: Glucose, BMI, Age
clustering_features = ['Glucose', 'BMI', 'Age']
X_cluster = df[clustering_features]

print(f"\nUsing features for clustering: {clustering_features}")

# Apply K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

print(f"\nCluster assignments: {np.unique(clusters, return_counts=True)}")

# Evaluate clustering quality
silhouette = silhouette_score(X_cluster, clusters)
calinski = calinski_harabasz_score(X_cluster, clusters)

print(f"\nCluster Quality Metrics: ")
print(f"  Silhouette Score: {silhouette:.4f} (closer to 1 is better)")
print(f"  Calinski-Harabasz Score: {calinski:.4f} (higher is better)")

# Calculate mean Glucose for each cluster
df['Cluster'] = clusters
cluster_means = df.groupby('Cluster')['Glucose'].mean()

print(f"\nCluster mean Glucose values:")
for cluster_id, mean_glucose in cluster_means.items():
    print(f" Cluster {cluster_id}: {mean_glucose:.4f}")

# Check glucose separation
separation = abs(cluster_means.iloc[1] - cluster_means.iloc[0])
print(f"  Glucose separation: {separation:.4f}")

# Identify diabetes cluster
diabetes_cluster = cluster_means.idxmax()
print(f"\nDiabetes cluster identified: Cluster {diabetes_cluster}")

# Analyze feature importance
print("\nFeature contribution to cluster separation:")
centers = kmeans.cluster_centers_
feature_diff = np.abs(centers[1] - centers[0])
for i, feature in enumerate(clustering_features):
    print(f"  {feature}: {feature_diff[i]:.4f}")

# Generate Outcome labels
df['Outcome'] = (df['Cluster'] == diabetes_cluster).astype(int)

# Check label distribution
outcome_counts = df['Outcome'].value_counts().sort_index()
print(f"\nOutcome label distribution:")
print(f"  0(No Diabetes): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.1f}%)")
print(f"  1(Diabetes): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.1f}%)")

# Remove temporary Cluster column
df_final = df.drop('Cluster', axis=1)

# Save data with labels
df_final.to_csv('diabetes_with_labels.csv', index=False)
print(f"\nFinal shape: {df_final.shape}")
print(f"Columns: {df_final.columns.tolist()}")
print("\nSaved: diabetes_with_labels.csv")

# Visualize clustering
plt.figure(figsize=(10, 6))
plt.scatter(df[df['Outcome']==0]['Glucose'],
            df[df['Outcome']==0]['BMI'],
            c = 'blue', label='No Diabetes', alpha=0.6)
plt.scatter(df[df['Outcome']==1]['Glucose'],
            df[df['Outcome']==1]['BMI'],
            c='red', label='Diabetes', alpha=0.6)
plt.xlabel('Glucose(normalized)')
plt.ylabel('BMI(normalized)')
plt.title('K-means Clustering: Diabetes vs No Diabetes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_clustering.png')
print("Saved visualization: kmeans_clustering.png")
plt.close()

