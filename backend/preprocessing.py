"""
preprocessing.py - Reproducible Preprocessing Pipeline & Feature Schema
Preserves exact data handling from existing Heart Disease ML project.
"""

from typing import Any, Dict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Exact feature names from heart_disease_uci.csv
NUMERICAL_FEATURES = [
    "age",
    "resting bps",
    "cholestrol",
    "mx heart rate achieved",
    "oldpeak",
    "major vessels visible(ca)",
]

CATEGORICAL_FEATURES = [
    "gender",
    "chest pain type",
    "fasting blood sugar",
    "restecg result",
    "exang",
    "slope",
    "thal",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"

# Mapping from common API/frontend naming conventions to exact dataset columns
FEATURE_NAME_MAPPING: Dict[str, str] = {
    # Age
    "age": "age",
    # Sex / Gender
    "gender": "gender",
    "sex": "gender",
    # Chest Pain Type
    "chest_pain_type": "chest pain type",
    "chest_pain": "chest pain type",
    "cp": "chest pain type",
    # Resting Blood Pressure
    "resting_bps": "resting bps",
    "resting_bp": "resting bps",
    "trestbps": "resting bps",
    # Serum Cholesterol
    "cholestrol": "cholestrol",
    "cholesterol": "cholestrol",
    "chol": "cholestrol",
    # Fasting Blood Sugar
    "fasting_blood_sugar": "fasting blood sugar",
    "fbs": "fasting blood sugar",
    # Resting ECG
    "restecg_result": "restecg result",
    "resting_ecg": "restecg result",
    "restecg": "restecg result",
    # Max Heart Rate
    "mx_heart_rate_achieved": "mx heart rate achieved",
    "max_heart_rate": "mx heart rate achieved",
    "thalach": "mx heart rate achieved",
    # Exercise Induced Angina
    "exang": "exang",
    "exercise_angina": "exang",
    # Oldpeak / ST depression
    "oldpeak": "oldpeak",
    # ST Slope
    "slope": "slope",
    "st_slope": "slope",
    # Major Vessels
    "major_vessels_visible": "major vessels visible(ca)",
    "major_vessels": "major vessels visible(ca)",
    "ca": "major vessels visible(ca)",
    # Thalassemia
    "thal": "thal",
    "thalassemia": "thal",
}

# Domain value normalization maps
GENDER_MAP = {
    "male": "Male",
    "m": "Male",
    "1": "Male",
    1: "Male",
    "female": "Female",
    "f": "Female",
    "0": "Female",
    0: "Female",
}

CHEST_PAIN_MAP = {
    "typical angina": "typical angina",
    "typical": "typical angina",
    "0": "typical angina",
    0: "typical angina",
    "atypical angina": "atypical angina",
    "atypical": "atypical angina",
    "1": "atypical angina",
    1: "atypical angina",
    "non-anginal": "non-anginal",
    "non anginal": "non-anginal",
    "2": "non-anginal",
    2: "non-anginal",
    "asymptomatic": "asymptomatic",
    "3": "asymptomatic",
    3: "asymptomatic",
}

BOOLEAN_MAP = {
    True: "TRUE",
    "true": "TRUE",
    "TRUE": "TRUE",
    "1": "TRUE",
    1: "TRUE",
    False: "FALSE",
    "false": "FALSE",
    "FALSE": "FALSE",
    "0": "FALSE",
    0: "FALSE",
}

RESTECG_MAP = {
    "normal": "normal",
    "0": "normal",
    0: "normal",
    "st-t abnormality": "st-t abnormality",
    "st-t": "st-t abnormality",
    "1": "st-t abnormality",
    1: "st-t abnormality",
    "lv hypertrophy": "lv hypertrophy",
    "hypertrophy": "lv hypertrophy",
    "2": "lv hypertrophy",
    2: "lv hypertrophy",
}

SLOPE_MAP = {
    "upsloping": "upsloping",
    "0": "upsloping",
    0: "upsloping",
    "flat": "flat",
    "1": "flat",
    1: "flat",
    "downsloping": "downsloping",
    "2": "downsloping",
    2: "downsloping",
}

THAL_MAP = {
    "normal": "normal",
    "0": "normal",
    0: "normal",
    "fixed defect": "fixed defect",
    "fixed": "fixed defect",
    "1": "fixed defect",
    1: "fixed defect",
    "reversable defect": "reversable defect",
    "reversible": "reversable defect",
    "reversable": "reversable defect",
    "2": "reversable defect",
    2: "reversable defect",
}


def normalize_patient_input(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms any incoming dictionary into a 1-row DataFrame
    with exact column names and normalized categorical representations.
    """
    normalized: Dict[str, Any] = {}

    for k, v in raw_input.items():
        mapped_key = FEATURE_NAME_MAPPING.get(k.lower().strip(), k)
        normalized[mapped_key] = v

    # Normalize categorical string formats
    if "gender" in normalized:
        val = normalized["gender"]
        normalized["gender"] = GENDER_MAP.get(val, str(val))

    if "chest pain type" in normalized:
        val = normalized["chest pain type"]
        normalized["chest pain type"] = CHEST_PAIN_MAP.get(val, str(val).lower())

    if "fasting blood sugar" in normalized:
        val = normalized["fasting blood sugar"]
        normalized["fasting blood sugar"] = BOOLEAN_MAP.get(val, str(val).upper())

    if "restecg result" in normalized:
        val = normalized["restecg result"]
        normalized["restecg result"] = RESTECG_MAP.get(val, str(val).lower())

    if "exang" in normalized:
        val = normalized["exang"]
        normalized["exang"] = BOOLEAN_MAP.get(val, str(val).upper())

    if "slope" in normalized:
        val = normalized["slope"]
        normalized["slope"] = SLOPE_MAP.get(val, str(val).lower())

    if "thal" in normalized:
        val = normalized["thal"]
        normalized["thal"] = THAL_MAP.get(val, str(val).lower())

    # Convert numeric fields
    for num_col in NUMERICAL_FEATURES:
        if num_col in normalized and normalized[num_col] is not None:
            try:
                normalized[num_col] = float(normalized[num_col])
            except (ValueError, TypeError):
                pass

    # Ensure all required features are present (missing features will be None, handled by imputer)
    df_data = {feat: [normalized.get(feat, None)] for feat in ALL_FEATURES}
    return pd.DataFrame(df_data)


def build_preprocessor(scale_numerical: bool = True) -> ColumnTransformer:
    """
    Constructs the Scikit-Learn ColumnTransformer matching the project's logic:
    - Numerical: SimpleImputer(mean) -> Optional StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OrdinalEncoder
    """
    num_steps = [("imputer", SimpleImputer(strategy="mean"))]
    if scale_numerical:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(classifier, scale_numerical: bool = True) -> Pipeline:
    """
    Combines the preprocessor and the classifier into a single, unified Scikit-Learn Pipeline.
    Inference only requires pipeline.predict(df) or pipeline.predict_proba(df).
    """
    preprocessor = build_preprocessor(scale_numerical=scale_numerical)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
