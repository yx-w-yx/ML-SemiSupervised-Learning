# Semi-Supervised Learning on Medical Datasets

Machine Learning Project - Northeastern University

## Overview

Implementation of Super Learner ensemble on two medical datasets using semi-supervised and supervised learning approaches.

**Datasets:**
- Diabetes: 733 → 725 rows, K-means generated labels
- Breast Cancer: 4024 → 3444 rows, real labels

**Methods:** Data preprocessing, K-means clustering, PCA, Super Learner (NB + KNN + NN → DT)

## Files

**Code:**
- `1_diabetes_preprocessing.py` - Clean diabetes data
- `2_cancer_preprocessing.py` - Clean cancer data  
- `3_diabetes_kmeans.py` - K-means clustering for labels
- `4_diabetes_modeling.py` - Diabetes modeling
- `5_cancer_modeling.py` - Cancer modeling
- `6_comparison.py` - Results comparison

**Data:**
- `diabetes_with_labels.csv` (725 × 8)
- `cancer_cleaned.csv` (3444 × 20)

**Results:**
- Visualizations and confusion matrices

## How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run in order
python 1_diabetes_preprocessing.py
python 2_cancer_preprocessing.py
python 3_diabetes_kmeans.py
python 4_diabetes_modeling.py
python 5_cancer_modeling.py
python 6_comparison.py
```

## Results

**Diabetes:** Test Accuracy =  0.8759

**Cancer:** Test Accuracy = 0.9013

**K-means Quality:** Silhouette Score = 0.45


