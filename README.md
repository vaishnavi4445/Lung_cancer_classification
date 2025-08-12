# 🫁 AI/ML Internship Task 6: Lung Cancer Classification Using KNN

This project is part of **Task 6** of my AI/ML internship, where I implemented a complete machine learning pipeline using the **K-Nearest Neighbors (KNN)** algorithm to classify lung cancer presence based on patient data. The project includes data preprocessing, model training, evaluation, and visualization, and is built to run seamlessly on **Google Colab**.

---

## 📊 Dataset Overview

- **Source**: [Kaggle – Lung Cancer Dataset](https://www.kaggle.com/datasets/yusufdede/lung-cancer-dataset)
- **Attributes**: Age, gender, smoking history, air pollution exposure, genetic risk, chronic lung disease, fatigue, allergy, wheezing, coughing, and more.
- **Target**: `LUNG_CANCER` (binary: `YES` or `NO`)

---

## 🧠 Objectives

- Clean and preprocess the dataset
- Normalize features for KNN
- Train and evaluate a KNN classifier
- Visualize performance using confusion matrix and PCA decision boundaries
- Save trained model for future use

---

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab
- Joblib (for model saving)

---

## 📈 Results

- Achieved strong classification performance
- Visualized decision boundaries using PCA
- Saved model and scaler for deployment

---

## 📁 Project Structure
lung-cancer-knn/
├── lung_cancer_knn.ipynb       # Main Google Colab notebook with full pipeline
├── knn_model.pkl               # Saved KNN model for reuse or deployment
├── scaler.pkl                  # Saved StandardScaler object for preprocessing
├── README.md                   # Project documentation and instructions
└── data/                       # Folder containing the dataset
    └── lung_cancer_example.csv         # Original dataset used for training and testing
---
