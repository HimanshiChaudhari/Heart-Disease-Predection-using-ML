"""
main.py - FastAPI Backend Service for Heart Disease Risk Prediction
Exposes REST endpoints for prediction, health check, model transparency, and static UI.
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from backend.predict import HeartDiseasePredictor
    from backend.preprocessing import ALL_FEATURES, FEATURE_NAME_MAPPING
    from backend.train_model import train_and_export
except ImportError:
    from predict import HeartDiseasePredictor
    from preprocessing import ALL_FEATURES, FEATURE_NAME_MAPPING
    from train_model import train_and_export

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-grade Machine Learning screening API based on the UCI Heart Disease dataset.",
    version="1.0.0",
)

# Enable CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
STATIC_DIR = os.path.join(CURRENT_DIR, "static")

# Ensure models are trained and present
def initialize_models():
    primary_model_path = os.path.join(MODEL_DIR, "heart_disease_pipeline.pkl")
    if not os.path.exists(primary_model_path):
        print("Model artifacts not found. Training and serializing models now...")
        workspace_dir = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
        candidate_csv = [
            os.path.join(workspace_dir, "heart_disease_uci.csv"),
            os.path.join(CURRENT_DIR, "heart_disease_uci.csv"),
            "heart_disease_uci.csv",
        ]
        csv_file = next((p for p in candidate_csv if os.path.exists(p)), None)
        if csv_file:
            train_and_export(dataset_path=csv_file, output_dir=MODEL_DIR)
        else:
            print("Warning: heart_disease_uci.csv not found for auto-training.")

initialize_models()
predictor = HeartDiseasePredictor(model_dir=MODEL_DIR)


# Pydantic Patient Input Request Schema
class PatientDataRequest(BaseModel):
    # Demographics
    age: float = Field(..., ge=1, le=120, description="Age in years", example=55)
    sex: Any = Field(..., description="Biological sex ('Male', 'Female', 1, 0)", example="Male")

    # Cardiac Symptoms
    chest_pain: Any = Field(
        ...,
        description="Chest pain type ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic')",
        example="asymptomatic",
    )

    # Vitals & Labs
    resting_bp: float = Field(..., ge=60, le=260, description="Resting blood pressure in mm Hg", example=135)
    cholesterol: float = Field(..., ge=80, le=700, description="Serum cholesterol in mg/dL", example=240)
    fasting_blood_sugar: Any = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dL ('TRUE', 'FALSE', True, False, 1, 0)",
        example="FALSE",
    )

    # Diagnostic Electrocardiogram & Stress Testing
    resting_ecg: Any = Field(
        ...,
        description="Resting ECG ('normal', 'st-t abnormality', 'lv hypertrophy')",
        example="normal",
    )
    max_heart_rate: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved", example=145)
    exercise_angina: Any = Field(
        ...,
        description="Exercise induced angina ('TRUE', 'FALSE', True, False, 1, 0)",
        example="TRUE",
    )
    oldpeak: float = Field(
        ...,
        ge=-5.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest",
        example=1.5,
    )
    st_slope: Any = Field(
        ...,
        description="Slope of peak exercise ST segment ('upsloping', 'flat', 'downsloping')",
        example="flat",
    )

    # Imaging Findings
    major_vessels: float = Field(
        0,
        ge=0,
        le=3,
        description="Number of major vessels (0-3) colored by fluoroscopy",
        example=1,
    )
    thalassemia: Any = Field(
        "normal",
        description="Thalassemia scan ('normal', 'fixed defect', 'reversable defect')",
        example="reversable defect",
    )

    # Optional model selector
    selected_model: Optional[str] = Field(
        "Logistic Regression",
        description="Model to use: 'Logistic Regression', 'Decision Tree', 'KNN (k=21)', 'Linear SVM', 'RBF SVM'",
        example="Logistic Regression",
    )


class RiskFactorItem(BaseModel):
    factor: str
    level: str
    description: str


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 for No Disease, 1 for Heart Disease Present")
    prediction_label: str = Field(..., description="Human-readable risk classification summary")
    probability: Optional[float] = Field(None, description="Predicted probability (0.0 to 1.0)")
    risk_level: str = Field(..., description="Categorized risk: 'Low', 'Moderate', 'High'")
    model: str = Field(..., description="Name of the Machine Learning model evaluated")
    risk_factors: List[RiskFactorItem] = Field(..., description="Clinical risk factor breakdown")
    disclaimer: str = Field(..., description="Mandatory medical screening disclaimer")


@app.get("/health", summary="API Health Check")
def health_check():
    """Returns server and model loading status."""
    primary_loaded = "Logistic Regression" in predictor.pipelines
    return {
        "status": "online",
        "service": "Heart Disease Risk Prediction API",
        "primary_model_loaded": primary_loaded,
        "available_models": list(predictor.metadata.get("models", {}).keys()),
    }


@app.get("/model-info", summary="Model Transparency & Dataset Metadata")
def model_info():
    """Returns comprehensive evaluation metrics, confusion matrices, and dataset statistics."""
    if not predictor.metadata:
        predictor._load_metadata()
    return predictor.metadata


@app.post("/predict", response_model=PredictResponse, summary="Predict Heart Disease Risk")
def predict_heart_disease(payload: PatientDataRequest):
    """
    Receives validated patient clinical parameters and returns prediction, probability,
    risk categorization, and transparent clinical explanations.
    """
    try:
        raw_dict = payload.dict()
        model_name = raw_dict.pop("selected_model", "Logistic Regression") or "Logistic Regression"
        result = predictor.predict(raw_patient_data=raw_dict, model_name=model_name)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=500, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/sample-patients", summary="Pre-configured Sample Patient Profiles")
def sample_patients():
    """Provides clinical presets for testing the application with 1 click."""
    return [
        {
            "profile_name": "Low Risk / Healthy Baseline",
            "description": "45-year-old female, normal BP, normal cholesterol, no angina, normal ECG and exercise response.",
            "data": {
                "age": 45,
                "sex": "Female",
                "chest_pain": "atypical angina",
                "resting_bp": 115,
                "cholesterol": 190,
                "fasting_blood_sugar": "FALSE",
                "resting_ecg": "normal",
                "max_heart_rate": 172,
                "exercise_angina": "FALSE",
                "oldpeak": 0.2,
                "st_slope": "upsloping",
                "major_vessels": 0,
                "thalassemia": "normal",
            },
        },
        {
            "profile_name": "Borderline / Moderate Risk",
            "description": "56-year-old male, slightly elevated BP and cholesterol, asymptomatic chest discomfort, mild ST depression.",
            "data": {
                "age": 56,
                "sex": "Male",
                "chest_pain": "non-anginal",
                "resting_bp": 138,
                "cholesterol": 245,
                "fasting_blood_sugar": "FALSE",
                "resting_ecg": "st-t abnormality",
                "max_heart_rate": 142,
                "exercise_angina": "FALSE",
                "oldpeak": 1.2,
                "st_slope": "flat",
                "major_vessels": 0,
                "thalassemia": "normal",
            },
        },
        {
            "profile_name": "High Risk / Coronary Artery Disease Indicated",
            "description": "63-year-old male, hypertensive, high cholesterol, asymptomatic chest pain, exercise angina, oldpeak 2.8mm, 2 vessels visible.",
            "data": {
                "age": 63,
                "sex": "Male",
                "chest_pain": "asymptomatic",
                "resting_bp": 155,
                "cholesterol": 286,
                "fasting_blood_sugar": "TRUE",
                "resting_ecg": "lv hypertrophy",
                "max_heart_rate": 115,
                "exercise_angina": "TRUE",
                "oldpeak": 2.8,
                "st_slope": "flat",
                "major_vessels": 2,
                "thalassemia": "reversable defect",
            },
        },
    ]


# Mount frontend distribution directory or static directory
FRONTEND_DIST = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend", "dist"))
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
elif os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
