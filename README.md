# Predicting Income Level with Machine Learning

**CSE422 – Artificial Intelligence | BRAC University | Spring 2025**

A machine learning project that predicts whether a person earns **more than $50K per year** using the U.S. Census *Adult Income* dataset. We clean and explore the data, then train and compare three classifiers: **K-Nearest Neighbours**, **Logistic Regression**, and a **Neural Network**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Train/Test Split](#traintest-split)
- [Models](#models)
- [Results](#results)
- [Key Findings](#key-findings)
- [Challenges](#challenges)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Team](#team)

---

## Overview

The goals of this project are to:

1. Build an accurate model that classifies income as `<=50K` or `>50K`.
2. Understand which factors (education, hours worked, capital gain, etc.) influence income the most.
3. Compare three models using **accuracy, precision, recall, F1-score, and ROC-AUC**.

## Dataset

| Property | Details |
|---|---|
| Source | U.S. Census – Adult Income dataset |
| Input features | 14 |
| Target | Income (`<=50K` / `>50K`) |
| Problem type | Binary classification |

**Numerical features:** Age, Final Weight, Education Number of Years, Capital-gain, Capital-loss, Hours-per-week

**Categorical features:** Workclass, Education, Marital-status, Occupation, Relationship, Race, Sex, Native-country

## Exploratory Data Analysis

- **Missing values:** Encoded as `"?"` in columns such as Workclass, Occupation, and Native-country.
- **Class imbalance:** About 75% of samples are `<=50K` and about 25% are `>50K`.
- **Correlation:** No strong correlation between numerical features (none above 0.85), so no features were dropped for multicollinearity. Education years, capital gain, and hours per week show weak-but-meaningful positive association with earning `>50K`.

## Preprocessing

| Step | Approach |
|---|---|
| Missing values | Replaced `"?"` with `NaN`; imputed categorical columns with the **most frequent** value and numerical columns with the **median** (no rows dropped) |
| Encoding | **Label Encoding** for binary columns, **One-Hot Encoding** for multi-class categorical columns |
| Scaling | **RobustScaler** (IQR-based, robust to outliers) applied to numeric columns only; encoded 0/1 columns left untouched |
| Class imbalance | Stratified splitting + class-weighted models |

## Train/Test Split

- **70%** train / **30%** test
- **Stratified random split** to preserve the original class ratio in both sets

## Models

- **K-Nearest Neighbours (KNN):** Simple, instance-based baseline using scaled features.
- **Logistic Regression:** Linear, interpretable classifier trained with `class_weight="balanced"`.
- **Neural Network:** Feed-forward network (TensorFlow/Keras) with 2 hidden layers, ReLU activation, dropout regularization, sigmoid output, and early stopping.

## Results

### Overall metrics

| Model | Accuracy | AUC | Precision (>50K) | Recall (>50K) | F1 (>50K) |
|---|---|---|---|---|---|
| KNN | **0.86** | 0.89 | **0.72** | 0.66 | **0.69** |
| Logistic Regression | 0.81 | **0.91** | 0.57 | **0.84** | 0.68 |
| Neural Network | 0.82 | 0.87 | 0.66 | 0.56 | 0.60 |

### Confusion matrices (test set, 14,653 samples)

| Model | True <=50K → Pred <=50K | False Positives | False Negatives | True >50K → Pred >50K |
|---|---|---|---|---|
| KNN | ~10,247 | 900 | 1,202 | 2,304 |
| Logistic Regression | 8,908 | 2,239 | 554 | 2,952 |
| Neural Network | ~10,131 | 1,016 | 1,551 | 1,955 |


## Key Findings

- **KNN** achieved the best overall accuracy (0.86) and the highest F1 on the `>50K` class.
- **Logistic Regression** achieved the best AUC (0.91) and the best recall on `>50K` (0.84), making it the strongest at ranking predictions and catching high earners, at the cost of more false positives.
- **Neural Network** was reasonably balanced but had lower recall on the high-income class.
- Proper preprocessing and choosing a model based on the metric that matters (accuracy vs. recall vs. AUC) had a big impact on results.

## Challenges

- Handling non-standard missing values (`"?"`)
- Class imbalance reducing recall on the minority class
- Scaling only numeric columns while preserving one-hot encoded features
- Training time and threshold tuning for the neural network

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## Getting Started


## Team

| Name | Section |
|---|---|
| Shahriar Mohammad | 15 |
| Nur Faisal Ahmed Sourove | 13 |

**Course:** CSE422 – Artificial Intelligence, BRAC University (Spring 2025)
**Instructors:** Pollock Nag [PLN] & Aunon Halder [CANH]
**Project demonstration:** 14 May 2025
