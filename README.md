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
- `step1_diabetes_preprocessing.py` - Clean diabetes data
- `step2_cancer_preprocessing.py` - Clean cancer data  
- `step3_diabetes_kmeans.py` - K-means clustering for labels
- `step4_diabetes_modeling.py` - Diabetes modeling
- `step5_cancer_modeling.py` - Cancer modeling
- `step6_comparison.py` - Results comparison

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
python step1_diabetes_preprocessing.py
python step2_cancer_preprocessing.py
python step3_diabetes_kmeans.py
python step4_diabetes_modeling.py
python step5_cancer_modeling.py
python step6_comparison.py
```

## Results

**Diabetes:** Test Accuracy = [your result]  
**Cancer:** Test Accuracy = [your result]

**K-means Quality:** Silhouette Score = 0.45

## Author

Yuexin Zhang - Northeastern University MGEN Program
