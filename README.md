# Toxic Comment Classification Challenge

This project is based on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge). The goal is to build a multi-label text classification model that detects different types of toxicity in online comments.

## 🧠 Problem Statement
Given a comment, classify it into one or more of the following categories:
- **toxic**
- **severe_toxic**
- **obscene**
- **threat**
- **insult**
- **identity_hate**

## 📂 Dataset
The dataset contains text comments extracted from Wikipedia’s talk page edits, with binary labels for each category of toxicity. The dataset is split into:
- `train.csv` — labeled data used for training
- `test.csv` — unlabeled data used for predictions

## ⚙️ Workflow
- Data exploration and analysis
- Text preprocessing (cleaning, stemming, stopwords removal)
- Feature extraction using TF-IDF
- Model training with:
  - Logistic Regression
  - Random Forest
  - Ridge Classifier
  - XGBoost (One-vs-Rest strategy)
- Model evaluation using F1-score

## 📦 Libraries Used
- `pandas`, `numpy`
- `sklearn`
- `xgboost`
- `nltk`

## 📊 Results
Various machine learning models were trained and evaluated using F1-score on cross-validation. The cleaned and vectorized comment text significantly improved performance.

## 📁 Structure
- `toxic_comment_classification.ipynb`: Main Jupyter notebook with full code and explanations
- `data/`: Contains train/test datasets (if included)
- `README.md`: Project description

## 📌 Notes
- NLTK resources like stopwords must be downloaded before preprocessing.
- The code supports Kaggle environments but can be modified for local execution.

## 🔗 Reference
- [Jigsaw Competition](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
