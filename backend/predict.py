"""
predict.py - Production Inference Engine for Heart Disease Risk Prediction
Loads unified Scikit-Learn pipelines and computes predictions, probabilities, and clinical factor insights.
"""

import json
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

try:
    from backend.preprocessing import ALL_FEATURES, normalize_patient_input
except ImportError:
    from preprocessing import ALL_FEATURES, normalize_patient_input


class HeartDiseasePredictor:
    """
    Production-ready prediction service.
    Loads Scikit-Learn unified pipeline (preprocessor + model) and provides
    calibrated predictions, probabilities, and explainability.
    """

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(current_dir, "model")
        else:
            self.model_dir = model_dir

        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        self.metadata: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.default_model_name = "Logistic Regression"

        self._load_metadata()
        self._load_default_pipeline()

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def _load_default_pipeline(self):
        default_file = os.path.join(self.model_dir, "heart_disease_pipeline.pkl")
        if os.path.exists(default_file):
            self.pipelines[self.default_model_name] = joblib.load(default_file)

    def get_pipeline(self, model_name: str = "Logistic Regression"):
        """Returns requested pipeline, loading from disk on demand."""
        if model_name in self.pipelines:
            return self.pipelines[model_name]

        # Look up filename in metadata
        model_meta = self.metadata.get("models", {}).get(model_name)
        filename = model_meta.get("filename") if model_meta else None

        if not filename:
            # Fallback mappings
            filename_map = {
                "Logistic Regression": "heart_disease_pipeline.pkl",
                "Decision Tree": "decision_tree_pipeline.pkl",
                "KNN (k=21)": "knn_pipeline.pkl",
                "KNN": "knn_pipeline.pkl",
                "Linear SVM": "svm_linear_pipeline.pkl",
                "RBF SVM": "svm_rbf_pipeline.pkl",
            }
            filename = filename_map.get(model_name, "heart_disease_pipeline.pkl")

        pipeline_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Trained pipeline artifact '{filename}' not found at {pipeline_path}")

        pipeline = joblib.load(pipeline_path)
        self.pipelines[model_name] = pipeline
        return pipeline

    def analyze_risk_factors(self, input_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyzes individual clinical parameters against known clinical risk thresholds
        and model feature weights to produce transparent patient risk explanations.
        """
        factors = []
        row = input_df.iloc[0]

        # 1. Chest pain
        cp = str(row.get("chest pain type", "")).lower()
        if cp == "asymptomatic":
            factors.append({
                "factor": "Chest Pain Type: Asymptomatic",
                "level": "High Risk Indicator",
                "description": "Silent ischemia / asymptomatic chest pain is strongly correlated with severe coronary disease in the UCI dataset."
            })
        elif cp in ["typical angina", "non-anginal"]:
            factors.append({
                "factor": f"Chest Pain: {cp.capitalize()}",
                "level": "Monitored",
                "description": f"Reported {cp} is documented and evaluated with other cardiac markers."
            })

        # 2. ST depression (Oldpeak)
        oldpeak = row.get("oldpeak")
        if oldpeak is not None and not pd.isna(oldpeak):
            if float(oldpeak) >= 2.0:
                factors.append({
                    "factor": f"ST Depression (Oldpeak: {oldpeak} mm)",
                    "level": "High Risk Indicator",
                    "description": "Significant ST depression during exercise (>= 2.0 mm) indicates potential myocardial ischemia."
                })
            elif float(oldpeak) >= 1.0:
                factors.append({
                    "factor": f"ST Depression (Oldpeak: {oldpeak} mm)",
                    "level": "Moderate Risk Indicator",
                    "description": "Mild to moderate exercise-induced ST depression."
                })

        # 3. Fluoroscopy major vessels
        ca = row.get("major vessels visible(ca)")
        if ca is not None and not pd.isna(ca):
            ca_val = float(ca)
            if ca_val > 0:
                factors.append({
                    "factor": f"Major Vessels Visible ({int(ca_val)})",
                    "level": "High Risk Indicator",
                    "description": f"{int(ca_val)} major coronary vessel(s) showed fluoroscopic narrowing/calcification."
                })

        # 4. Thalassemia defect
        thal = str(row.get("thal", "")).lower()
        if thal in ["reversable defect", "fixed defect"]:
            factors.append({
                "factor": f"Thallium Stress Defect: {thal.capitalize()}",
                "level": "High Risk Indicator",
                "description": "Presence of fixed or reversible perfusion defect on thallium cardiac imaging."
            })

        # 5. Exercise Induced Angina
        exang = str(row.get("exang", "")).upper()
        if exang == "TRUE":
            factors.append({
                "factor": "Exercise-Induced Angina Present",
                "level": "Moderate to High Risk",
                "description": "Chest discomfort triggered by exertion suggests compromised cardiac blood supply."
            })

        # 6. Resting Blood Pressure
        rbp = row.get("resting bps")
        if rbp is not None and not pd.isna(rbp):
            rbp_val = float(rbp)
            if rbp_val >= 140:
                factors.append({
                    "factor": f"Elevated Resting Blood Pressure ({int(rbp_val)} mmHg)",
                    "level": "Cardiovascular Strain",
                    "description": "Systolic blood pressure >= 140 mmHg qualifies as Stage 2 Hypertension."
                })

        # 7. Cholesterol
        chol = row.get("cholestrol")
        if chol is not None and not pd.isna(chol):
            chol_val = float(chol)
            if chol_val >= 240:
                factors.append({
                    "factor": f"High Serum Cholesterol ({int(chol_val)} mg/dL)",
                    "level": "Lipid Risk Factor",
                    "description": "Serum cholesterol >= 240 mg/dL increases long-term risk of arterial plaque buildup."
                })

        return factors

    def predict(
        self,
        raw_patient_data: Dict[str, Any],
        model_name: str = "Logistic Regression",
    ) -> Dict[str, Any]:
        """
        Executes end-to-end prediction:
        1. Normalizes input dictionary into expected DataFrame schema
        2. Queries the Scikit-Learn unified pipeline
        3. Returns structured prediction, probability, risk level, and clinical explanations
        """
        # 1. Normalize input
        input_df = normalize_patient_input(raw_patient_data)

        # 2. Get requested pipeline
        pipeline = self.get_pipeline(model_name)

        # 3. Compute prediction
        pred = int(pipeline.predict(input_df)[0])

        # 4. Compute probability if model supports it
        prob = None
        has_proba = hasattr(pipeline, "predict_proba")
        if has_proba:
            try:
                proba_array = pipeline.predict_proba(input_df)[0]
                prob = round(float(proba_array[1]), 4)
            except Exception:
                prob = None

        # 5. Determine qualitative risk level and label
        if prob is not None:
            if prob >= 0.50:
                risk_level = "High"
                pred_label = "Higher predicted risk"
            else:
                risk_level = "Low"
                pred_label = "Lower predicted risk according to this model"
        else:
            risk_level = "High" if pred == 1 else "Low"
            pred_label = "Higher predicted risk" if pred == 1 else "Lower predicted risk according to this model"

        # 6. Clinical factor breakdown
        risk_factors = self.analyze_risk_factors(input_df)

        return {
            "prediction": pred,
            "prediction_label": pred_label,
            "probability": prob,
            "risk_level": risk_level,
            "model": model_name,
            "risk_factors": risk_factors,
            "disclaimer": (
                "This ML-based result is for educational and screening purposes only. "
                "It is not a medical diagnosis and should not replace professional medical evaluation. "
                "If you have symptoms or concerns about your health, consult a qualified healthcare professional."
            ),
        }
