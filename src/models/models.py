import pandas as pd
import joblib
import argparse
import os
import torch

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from tabpfn import TabPFNClassifier

''' 
Sample command line usage:
# Train TabPFN model
python src/models/models.py --model tabpfn --eval true

# config for TabPFN to be included in ensemble
python src/models/ensemble.py --substitute_model path/to/tabpfn_model.pkl 

'''
parser = argparse.ArgumentParser(description="train and optionally evaluate model")
parser.add_argument('--model', type=str, default=None, choices=["logistic_regression", "tabpfn", "ensemble" ], help="choose model to train")
parser.add_argument('--substitute_model', type=str, default=None, choices=["./tabpfn.pkl"], help="choose base model to use instead of XGBoost for building ensemble model")
parser.add_argument('--eval', type=str, default=None, choices=["true", "false"], help="set true to evaluate model with test data")
args = parser.parse_args()

# Define helper functions
def has_gpu():
    return torch.cuda.is_available()

def save_model(model, name: str):
    path = f"src/models/{name}.pkl"
    joblib.dump(model, path)
    print(f"{name} model saved to {path}")

def eval_model(model):
    print(f"Evaluating {model} model...")
    if model == "ensemble":
        y_pred = stacked_clf.predict(X_test)
    elif model == "logistic_regression":
        y_pred = lr_model.predict(X_test)
    elif model == "tabpfn":
        y_pred = tabpfn_model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
    print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# === Baseline Logistic Regression ===
if args.model == "logistic_regression":
    print("Training baseline Logistic Regression")
    # === Load & preprocess data ===
    df = pd.read_csv("data/interim/cleaned_data.csv")
    target_col = "fraud_reported"

    X = df.drop(target_col, axis=1)
    X = X.drop(columns=X.select_dtypes(include=['object', 'string', 'category']).columns)
    y = df['fraud_reported'].map({"Y":1,"N":0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Train model ===
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    print(f"{args.model} Trained")

    # === Save model ===
    save_model(lr_model, args.model)

# === Train Tabular Prior-data Fitted Networks ===
elif args.model == "tabpfn":
    print("Training tabpfn model")
    # === Load & preprocess data ===
    df = pd.read_csv("data/processed/processed_data.csv")
    target_col = "fraud_reported"
    boolean_cols = df.select_dtypes(include=[bool]).columns # Convert bool columns to int because tabpfn expects numeric values
    df[boolean_cols] = df[boolean_cols].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # === Train model ===
    tabpfn_model = TabPFNClassifier(device="cuda" if has_gpu() else "cpu")
    tabpfn_model.fit(X_train, y_train)
    print(f"{args.model} Trained")

    # === Save Model ===
    save_model(tabpfn_model, args.model)

# === Ensemble ===
elif args.model == "ensemble":
    print("Training ensemble model")
    # === Load & preprocess data ===
    df = pd.read_csv("data/processed/processed_data.csv")
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

    # === Save Final Model ===
    save_model(stacked_clf, args.model)

    # === Evaluation ===
    y_pred = stacked_clf.predict(X_test)
    y_prob = stacked_clf.predict_proba(X_test)[:, 1] # not used?

# === Evaluate Model ===
if args.eval == "true":
    eval_model(args.model)
