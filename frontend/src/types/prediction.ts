export interface PatientInput {
  age: number;
  sex: string | number;
  chest_pain: string;
  resting_bp: number;
  cholesterol: number;
  fasting_blood_sugar: string | boolean;
  resting_ecg: string;
  max_heart_rate: number;
  exercise_angina: string | boolean;
  oldpeak: number;
  st_slope: string;
  major_vessels: number;
  thalassemia: string;
  selected_model?: string;
}

export interface RiskFactorItem {
  factor: string;
  level: string;
  description: string;
}

export interface PredictionResult {
  prediction: number;
  prediction_label: string;
  probability: number | null;
  risk_level: 'Low' | 'Moderate' | 'High';
  model: string;
  risk_factors: RiskFactorItem[];
  disclaimer: string;
}

export interface ModelMetric {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number | null;
  filename: string;
  description: string;
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
    matrix: number[][];
  };
}

export interface ModelMetadata {
  dataset: {
    filename: string;
    total_samples: number;
    train_samples: number;
    test_samples: number;
    target_definition: string;
    numerical_features: string[];
    categorical_features: string[];
    all_features: string[];
  };
  models: Record<string, ModelMetric>;
  primary_production_model: string;
  primary_pipeline_file: string;
  feature_coefficients_log_odds?: Record<string, number>;
  disclaimer: string;
}

export interface PatientPreset {
  profile_name: string;
  description: string;
  data: PatientInput;
}
