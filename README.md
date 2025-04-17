# 🕵️‍♂️ DSA4263 ScamSnipers: Machine Learning in Fraudulent Insurance Claims 

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributors](https://img.shields.io/badge/contributors-6-orange)


This project provides a comprehensive solution for detecting insurance fraud claims ([Kaggle](https://www.kaggle.com/code/buntyshah/insurance-fraud-claims-detection)) using machine learning techniques. The approach covers the data science pipeline, from initial data analysis and exploration to feature engineering and model development. The focus is to build a robust fraud detection system, to reduce operational costs and enhance decision-making processes within insurance companies.

---

## 📑 Table of Contents  

- [Introduction](#introduction)  
- [Project Structure](#project-structure)
- [Dataset](#dataset)  
- [Methods](#methods)  
- [Models](#models)  
- [Results](#results)  
- [Use Case](#use-case)  
- [Conclusion](#conclusion)  
- [How to Run](#how-to-run)  
- [Contributors](#contributors)  
- [References](#references)  

---

## 📌 Introduction  

Insurance fraud, particularly in automobile claims, leads to significant financial losses. In Singapore alone, approximately **20% of motor insurance claims involve fraud** (GIA, 2018). This project aims to leverage machine learning models to classify claims as fraudulent or legitimate, offering decision support in insurance operations.

---

## Project Structure

```plaintext
├── LICENSE                   # License for the project
├── Makefile                  # Makefile for managing essential workflows
├── README.md                 # Project documentation
│
├── data                      # Data storage folder
│   ├── interim               # Intermediate data after data cleaning
│   ├── processed             # Final data for modeling after feature engineering
│   └── raw                   # The original, immutable data dump
│
├── .env.template             # Template for .env file
│
├── docs                      # Sphinx documentation
│
├── notebook                  # Jupyter notebook for entire pipeline
│
├── references.txt            # Data dictionaries, manuals, and other explanatory materials
│
├── reports                   # Generated analysis reports as HTML, PDF, LaTeX, etc.
│   ├── report.md             # Markdown to display visualizations
│   └── figures               # Figures and graphics for reporting
│
├── requirements.txt          # Python dependencies for project
│
├── setup.py                  # Makes project installable with `pip install -e .`
│
├── src                       # Source code
│   ├── data                  # Scripts to download or generate data
│   │   ├── make_dataset.py 
│   │   ├── clean.py 
│   │   ├── datadictionary.txt 
│   ├── features              # Scripts to build features from raw data
│   │   ├── build_features.py 
│   ├── models                # Scripts to train and predict with models
│   │   └── models.py 
│   └── visualization         # Scripts to generate visualizations
│       └── visualize.py
│
└── tox.ini                   # Configuration file for running `tox` tests

```
---

## 📊 Dataset  

- **Name:** `insurance_claims.csv`  
- **Size:** 1,000 rows × 40 columns  
- **Target Variable:** `fraud_reported` (Y/N)  
- **Key Features:**  
  - `policy_annual_premium`, `total_claim_amount`, `incident_type`, `insured_occupation`, `insured_hobbies`  

Preprocessing steps included:
- Handling missing values
- Grouping continuous variables (age ranges, tenure bins)
- Encoding categorical features (One-hot, Ordinal)

---

## 🛠️ Methods  

- **Exploratory Data Analysis (EDA)** to validate hypotheses about fraud patterns  
- **Feature Engineering:** Created derived features like `incident_minus_policy_bind_date`  
- **Sampling Methods:** SMOTE, ADASYN, and Random Undersampling  
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 5-fold cross-validation  

---

## 🤖 Models  

1. **Baseline:** Logistic Regression  
2. **Challenger 1:** Random Forest  
3. **Challenger 2:** TabPFN (transformer-based tabular model)  
4. **Ensemble Model:**  
   - **Stacking:** Random Forest + Gradient Boosting + TabPFN  
   - **Similarity Matching:** Localized KNN-based detection by incident state  

---

## 📈 Results  

| Model                        | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----------------------------|:----------|:-----------|:--------|:----------|:-----------|
| Logistic Regression          | 0.73     | 1.00      | 0.018   | 0.036    | 0.509    |
| Random Forest (ADASYN)       | 0.79     | 0.616     | 0.352   | 0.445    | 0.641    |
| TabPFN                       | 0.85     | 0.654     | 0.887   | 0.751    | 0.865    |
| **Ensemble Model**           | **0.864**| **0.673** | **0.879**| **0.762**| **0.897**|

---

## 🏛️ Use Case  

This fraud detection system can be integrated into an insurance company’s workflow:

1. Data ingestion from real-time and historical sources.
2. Preprocessing and feature engineering.
3. Model scoring with fraud probability and explanation (SHAP).
4. Flagging of high-risk claims for investigator review.
5. Feedback loop for continuous model retraining.

---

## 📌 Conclusion  

The ensemble model combining stacking and similarity matching achieved the best performance. It provides a scalable, interpretable, and industry-relevant approach for fraud detection in insurance.

---

## 🖥️ How to Run  

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 👨‍💻 Contributors

Choy Zhen Wen Marcus

Chew Yu Cai

Freedy Tan Wei You

Lau Mei Jin

Lim Swee En

Low Jia Li Natalie
