"""
train_model.py - Refactored Production Model Training & Evaluation
Fits Scikit-Learn unified pipelines on heart_disease_uci.csv and exports artifacts.
"""

import json
import os
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from backend.preprocessing import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        build_full_pipeline,
    )
except ImportError:
    from preprocessing import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        build_full_pipeline,
    )


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Loads dataset, drops id, and applies binary target transformation."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    data = pd.read_csv(csv_path)

    # 1. Drop id column as in original project
    if "id" in data.columns:
        data = data.drop("id", axis=1)

    # 2. Binary target transformation as in original project (line 20):
    # data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)

    return data


def train_and_export(
    dataset_path: str = "heart_disease_uci.csv",
    output_dir: str = "model",
) -> Dict[str, Any]:
    """
    Trains all existing models from the project using unified pipelines,
    evaluates them on the 30% test split, and exports the artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {dataset_path}")
    df = load_and_prepare_data(dataset_path)

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Train / Test split matching original project (70% train / 30% test, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Class distribution in test set - Negative: {sum(y_test == 0)}, Positive: {sum(y_test == 1)}")

    # Model definitions matching original project parameters
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "filename": "heart_disease_pipeline.pkl",  # Primary production model
            "description": "Logistic Regression with calibrated probabilities and feature log-odds explainability",
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                criterion="entropy",
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
            "filename": "decision_tree_pipeline.pkl",
            "description": "Decision Tree (entropy, max_depth=5, balanced class weights)",
        },
        "KNN (k=21)": {
            "model": KNeighborsClassifier(n_neighbors=21),
            "filename": "knn_pipeline.pkl",
            "description": "K-Nearest Neighbors (k=21)",
        },
        "Linear SVM": {
            "model": SVC(kernel="linear", probability=True, random_state=42),
            "filename": "svm_linear_pipeline.pkl",
            "description": "Support Vector Classifier with Linear Kernel",
        },
        "RBF SVM": {
            "model": SVC(kernel="rbf", probability=True, random_state=42),
            "filename": "svm_rbf_pipeline.pkl",
            "description": "Support Vector Classifier with Radial Basis Function (RBF) Kernel",
        },
    }

    metrics_summary: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Any] = {}

    for name, config in models_config.items():
        print(f"\n--- Training {name} ---")
        pipeline = build_full_pipeline(config["model"], scale_numerical=True)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        has_proba = hasattr(pipeline, "predict_proba")

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        roc_auc = None
        if has_proba:
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_proba))
            except Exception:
                roc_auc = None

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "matrix": cm.tolist(),
            },
            "filename": config["filename"],
            "description": config["description"],
        }

        # Save fitted pipeline using joblib
        save_path = os.path.join(output_dir, config["filename"])
        joblib.dump(pipeline, save_path)
        print(f"Saved pipeline to: {save_path}")
        print(
            f"Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc if roc_auc else 'N/A'}"
        )
        fitted_pipelines[name] = pipeline

    # Extract Logistic Regression coefficients for feature explainability
    lr_pipeline = fitted_pipelines["Logistic Regression"]
    lr_model = lr_pipeline.named_steps["classifier"]
    feature_coefficients = {}
    if hasattr(lr_model, "coef_"):
        raw_coefs = lr_model.coef_[0].tolist()
        for idx, feat in enumerate(ALL_FEATURES):
            if idx < len(raw_coefs):
                feature_coefficients[feat] = round(raw_coefs[idx], 4)

    # Complete metadata dictionary
    metadata = {
        "dataset": {
            "filename": "heart_disease_uci.csv",
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "target_definition": "0: No Heart Disease, 1: Heart Disease Present (converted via target > 0)",
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "all_features": ALL_FEATURES,
        },
        "models": metrics_summary,
        "primary_production_model": "Logistic Regression",
        "primary_pipeline_file": "heart_disease_pipeline.pkl",
        "feature_coefficients_log_odds": feature_coefficients,
        "disclaimer": (
            "This application is a Machine Learning screening and educational demonstration project. "
            "It is NOT medically validated and must NOT be used as a substitute for professional medical diagnosis, "
            "clinical advice, or treatment."
        ),
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    # Resolve paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(current_dir, ".."))

    candidate_csv_paths = [
        os.path.join(workspace_dir, "heart_disease_uci.csv"),
        os.path.join(current_dir, "heart_disease_uci.csv"),
        "heart_disease_uci.csv",
    ]

    selected_csv = next((p for p in candidate_csv_paths if os.path.exists(p)), None)
    if not selected_csv:
        raise FileNotFoundError("heart_disease_uci.csv could not be found.")

    output_model_dir = os.path.join(current_dir, "model")
    train_and_export(selected_csv, output_model_dir)
