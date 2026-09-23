# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced dataset.

## Overview
This project builds and compares classification models to identify fraudulent transactions among 15,936 records, where only 0.46% are fraud cases. The severe class imbalance is addressed using SMOTE (Synthetic Minority Oversampling Technique), and multiple models are evaluated beyond simple accuracy to properly assess minority-class performance.

## Pipeline
1. **Exploratory Data Analysis** — inspected class distribution, transaction amount/time patterns, and missing values.
2. **Feature Engineering** — log-transformed and standardized `Amount` and `Time` features.
3. **Baseline Modeling** — trained a Decision Tree classifier and evaluated with a confusion matrix.
4. **Class Imbalance Handling** — applied SMOTE to oversample the minority (fraud) class.
5. **Model Comparison** — trained and evaluated Random Forest and XGBoost classifiers.

## Results

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|
| Random Forest | 99.87% | 0.90 | 0.90 | 0.90 |
| Decision Tree + SMOTE | 99.84% | 0.86 | 0.90 | 0.88 |
| XGBoost | 99.87% | 0.86 | 0.95 | 0.90 |

XGBoost achieved the best balance of precision and recall on the minority fraud class, which is the primary goal in fraud detection (minimizing missed fraud cases).

## Tech Stack
Python, Pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, Matplotlib

## Dataset
Credit card transaction dataset with anonymized PCA-transformed features (V1–V28), transaction `Amount`, `Time`, and binary `Class` label (0 = non-fraud, 1 = fraud).
