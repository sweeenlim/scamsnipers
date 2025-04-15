import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import argparse
import os

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, confusion_matrix

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# === Argparse ==
# config for TabPFN to be included
# python src/models/ensemble.py --substitute_model path/to/tabpfn_model.pkl
parser = argparse.ArgumentParser(description="Train ensemble model with model substitution")
parser.add_argument('--model', type=str, default=None, help="Which model to train and run")
parser.add_argument('--substitute_model', type=str, default=None, help="Path to a pre-trained model to use instead of XGBoost")
args = parser.parse_args()

# === Baseline Logistic Regression ===
if args.model == "baseline_logistic_regression":
    print("Training baseline Logistic Regression")
    df = pd.read_csv("data/interim/cleaned_data.csv")

    target_col = "fraud_reported"

    X = df.drop(target_col, axis=1)
    X = X.drop(columns=X.select_dtypes(include=['object', 'string', 'category']).columns)

    y = df['fraud_reported'].map({"Y":1,"N":0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score: {:.4f}".format(roc_auc_score(y_test, y_pred)))
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# === Ensemble ===
if args.model == "ensemble":
    print("Training ensemble model")
    # === Load & preprocess data ===
    df = pd.read_csv("data/processed/processed_df.csv")
    target_col = "fraud_reported"

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Done Cleaning Data")

    # === Train Random Forest ===
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions={
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print("Random Forest Trained")

    # === Train Gradient Boosting ===
    gb_search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions={
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
    )
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    print("Gradient Boosting Trained")

    # === Load or Train XGBoost or Substitute Model ===
    if args.substitute_model and os.path.exists(args.substitute_model):
        print(f"Loading substitute model from {args.substitute_model}")
        best_substitute = joblib.load(args.substitute_model)
        model_name = os.path.splitext(os.path.basename(args.substitute_model))[0]
        print("Sub Model/ TabPFN Loaded")
    else:
        print("Training XGBoost model")
        xgb_search = RandomizedSearchCV(
            estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            param_distributions={
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
        )
        xgb_search.fit(X_train, y_train)
        best_substitute = xgb_search.best_estimator_
        model_name = 'xgb'
        print("XGBoost Trained")

    # === Stacking Classifier ===
    estimators = [('rf', best_rf), ('gb', best_gb), (model_name, best_substitute)]
    meta_learner = SVC(kernel='linear', probability=True, random_state=42)

    stacked_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    print("Ensemble Built")

    stacked_clf.fit(X_train, y_train)

    # === Evaluation ===
    y_pred = stacked_clf.predict(X_test)
    y_prob = stacked_clf.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score: {:.4f}".format(roc_auc_score(y_test, y_prob)))
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # === Save Final Model ===
    path = "src/models/ensemble_model.pkl"
    joblib.dump(stacked_clf, path)
    print(f"Model saved to {path}")
